"""Calibration (Constrained Risk Selection) and Frozen Held-out Evaluation Runner (Stage 4)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_CACHE_DIR, DATA_RAW_DIR, EVAL_DIR
from src.generator import ask_policy
from src.indexer import build_or_load_index, get_genai_client
from src.models import RetrievedCandidate
from src.retriever import HybridRetriever, _check_lexical_code_rescue
from src.verifier import extract_canonical_numbers

CALIBRATION_PATH = EVAL_DIR / "calibration_set.json"
HELDOUT_PATH = EVAL_DIR / "heldout_test_set.json"
CHECKPOINT_PATH = EVAL_DIR / "eval_checkpoint_progress.json"
RESULTS_JSON_PATH = EVAL_DIR / "eval_results.json"
REPORT_MD_PATH = EVAL_DIR / "eval_report.md"

_SLICE_LABELS: dict[str, str] = {
    "accented": "Có dấu",
    "unaccented": "Không dấu",
    "multi_turn": "Đa lượt",
    "near_miss_injection": "Near-miss & Injection",
}


def _normalize_match_text(text: str) -> str:
    s = unicodedata.normalize("NFC", text).lower()
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def check_must_contain(answer: str, must_contain: list[str]) -> bool:
    """Return True if answer satisfies all required numbers, percentages, or conditions."""
    if not must_contain:
        return True
    norm_ans = _normalize_match_text(answer)
    ans_nums = extract_canonical_numbers(answer, include_unscaled=True)

    for raw_req in must_contain:
        req = _normalize_match_text(raw_req)
        if not req:
            continue
        if req in norm_ans:
            continue

        # Handle date like "05/6" matching "5/6"
        if re.fullmatch(r"\d{1,2}/\d{1,2}", req):
            d_part, m_part = req.split("/")
            alt_date = f"{int(d_part)}/{int(m_part)}"
            if alt_date in norm_ans:
                continue

        # Handle percentage like "50%" matching "50 phần trăm" or "50 %"
        pct_m = re.fullmatch(r"(\d+(?:\.\d+)?)%", req)
        if pct_m:
            num_str = pct_m.group(1)
            if f"{num_str} %" in norm_ans or f"{num_str} phần trăm" in norm_ans:
                continue
            return False

        # Handle pure numeric tokens (e.g., "09" <-> "9", "1,5" <-> "1.5")
        if re.fullmatch(r"\d+(?:\.\d+)?", req):
            req_nums = extract_canonical_numbers(req, include_unscaled=True)
            if req_nums and req_nums.issubset(ans_nums):
                continue
            return False

        return False
    return True


def _passes_gate_at_tau(
    rec: dict[str, Any],
    tau_reject: float,
    tau_floor: float,
    candidates: list[RetrievedCandidate] | None = None,
) -> bool:
    top1_cos = float(rec["top1_cosine"])
    if top1_cos >= tau_reject:
        return True

    if candidates is not None:
        raw_queries = [str(rec["query"]), str(rec.get("query_used", rec["query"]))]
        if _check_lexical_code_rescue(raw_queries, candidates, top1_cos, tau_floor):
            return True
    elif bool(rec.get("lexical_code_match")) and top1_cos >= tau_floor * 0.85:
        return True

    bm25_top1 = float(rec["bm25_top1_score"])
    bm25_z = float(rec["bm25_zscore"])
    return top1_cos >= tau_floor and bm25_top1 >= 28.0 and bm25_z >= 5.0


def select_calibrated_threshold(
    calib_records: list[dict[str, Any]],
    tau_grid: list[float] | None = None,
    max_far: float = 0.05,
) -> dict[str, Any]:
    """Constrained Risk Selection: pick smallest tau satisfying FAR <= max_far (0/20 preferred)."""
    grid = tau_grid or [round(0.50 + 0.01 * i, 2) for i in range(36)]
    pos_items = [r for r in calib_records if r["is_positive"]]
    neg_items = [r for r in calib_records if not r["is_positive"]]
    n_pos = max(len(pos_items), 1)
    n_neg = max(len(neg_items), 1)

    sweep: list[dict[str, Any]] = []
    for tau in grid:
        tau_floor = round(max(0.50, tau - 0.07), 2)
        pos_pass = sum(
            1 for r in pos_items if _passes_gate_at_tau(r, tau, tau_floor, r.get("candidates"))
        )
        neg_pass = sum(
            1 for r in neg_items if _passes_gate_at_tau(r, tau, tau_floor, r.get("candidates"))
        )
        recall = pos_pass / n_pos
        far = neg_pass / n_neg
        sweep.append(
            {
                "tau_reject": tau,
                "tau_floor": tau_floor,
                "pos_pass": pos_pass,
                "pos_total": len(pos_items),
                "neg_pass": neg_pass,
                "neg_total": len(neg_items),
                "pos_recall": round(recall, 4),
                "neg_far": round(far, 4),
            }
        )

    zero_far = [row for row in sweep if row["neg_far"] == 0.0 and row["pos_recall"] >= 0.85]
    if zero_far:
        chosen = min(zero_far, key=lambda r: float(r["tau_reject"]))
    else:
        feasible = [row for row in sweep if float(row["neg_far"]) <= max_far]
        if feasible:
            # Smallest tau within max_far that preserves highest recall
            best_recall = max(float(r["pos_recall"]) for r in feasible)
            top_feasible = [r for r in feasible if float(r["pos_recall"]) >= best_recall - 1e-6]
            chosen = min(top_feasible, key=lambda r: float(r["tau_reject"]))
        else:
            chosen = min(sweep, key=lambda r: (float(r["neg_far"]), -float(r["pos_recall"]), float(r["tau_reject"])))

    return {
        "chosen": chosen,
        "sweep": sweep,
    }


def _candidate_matches_gold(
    doc_id: str,
    section_code: str,
    gold_doc_id: str | None,
    acceptable_clauses: list[str],
) -> bool:
    if not gold_doc_id or doc_id != gold_doc_id:
        return False
    if not acceptable_clauses:
        return True
    return section_code in acceptable_clauses


def _compute_group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pos_rows = [r for r in rows if r["expected_action"] == "ANSWERED"]
    neg_rows = [r for r in rows if r["expected_action"] != "ANSWERED"]

    n_pos = len(pos_rows)
    n_neg = len(neg_rows)

    hit_at_5 = (sum(1 for r in pos_rows if r["retrieval_hit"]) / n_pos) if n_pos else None
    verified_cite = (sum(1 for r in pos_rows if r["verified_citation_ok"]) / n_pos) if n_pos else None
    answer_corr = (sum(1 for r in pos_rows if r["answer_correct"]) / n_pos) if n_pos else None
    frr = (sum(1 for r in pos_rows if r["frr_error"]) / n_pos) if n_pos else None
    far = (sum(1 for r in neg_rows if r["far_error"]) / n_neg) if n_neg else None
    action_acc = (sum(1 for r in neg_rows if r["action_exact_match"]) / n_neg) if n_neg else None

    latencies = [float(r["latency_ms"]) for r in rows]
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    avg_calls = float(np.mean([int(r["api_calls"]) for r in rows])) if rows else 0.0

    return {
        "n_total": len(rows),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "retrieval_hit_at_5": round(hit_at_5, 4) if hit_at_5 is not None else None,
        "verified_citation_accuracy": round(verified_cite, 4) if verified_cite is not None else None,
        "answer_correctness": round(answer_corr, 4) if answer_corr is not None else None,
        "far": round(far, 4) if far is not None else None,
        "frr": round(frr, 4) if frr is not None else None,
        "blocked_action_accuracy": round(action_acc, 4) if action_acc is not None else None,
        "latency_p50_ms": round(p50, 1),
        "latency_p95_ms": round(p95, 1),
        "avg_api_calls_per_turn": round(avg_calls, 2),
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"calib": {}, "heldout": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {"calib": data.get("calib", {}), "heldout": data.get("heldout", {})}
    except (OSError, json.JSONDecodeError):
        pass
    return {"calib": {}, "heldout": {}}


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _fmt_pct(val: float | None) -> str:
    return f"{val * 100:.1f}%" if val is not None else "N/A"


def _render_markdown_report(summary: dict[str, Any]) -> str:
    cal = summary["calibration"]
    chosen = cal["chosen"]
    overall = summary["overall"]
    slices = summary["slices"]

    lines = [
        "# Báo cáo Thực nghiệm Giai đoạn 4 — Crema HR & HSE Policy RAG",
        "",
        "## 1. Hiệu chuẩn Ngưỡng Rủi ro Bất đối xứng (Calibration Set — 36 câu)",
        f"- **Ngưỡng được chọn:** `TAU_COSINE_REJECT = {chosen['tau_reject']:.2f}`, `TAU_COSINE_FLOOR = {chosen['tau_floor']:.2f}`",
        (
            f"- **Calibration Positive Recall (Gate):** `{chosen['pos_pass']}/{chosen['pos_total']}` "
            f"(`{chosen['pos_recall'] * 100:.1f}%`) | Retrieval Hit@5: `{cal['pos_hit_at_5'] * 100:.1f}%`"
        ),
        (
            f"- **Calibration Negative FAR (Gate):** `{chosen['neg_pass']}/{chosen['neg_total']}` "
            f"(`{chosen['neg_far'] * 100:.1f}%`, ràng buộc `<= 5.0%`)"
        ),
        "",
        "## 2. Kết quả Độc lập trên Tập Kiểm chuẩn Giữ riêng (Held-out Test Set — 60 câu)",
        "",
        "| Lát cắt | Số câu (Dương/Âm) | Hit@5 | Verified Cite | Answer Correctness | FAR | FRR | P50 (ms) | P95 (ms) | API/Turn |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    def add_row(label: str, m: dict[str, Any]) -> None:
        lines.append(
            f"| {label} | {m['n_total']} ({m['n_positive']}/{m['n_negative']}) | "
            f"{_fmt_pct(m['retrieval_hit_at_5'])} | {_fmt_pct(m['verified_citation_accuracy'])} | "
            f"{_fmt_pct(m['answer_correctness'])} | {_fmt_pct(m['far'])} | {_fmt_pct(m['frr'])} | "
            f"{m['latency_p50_ms']:.0f} | {m['latency_p95_ms']:.0f} | {m['avg_api_calls_per_turn']:.2f} |"
        )

    add_row("**Tổng thể (Held-out)**", overall)
    for key in ["accented", "unaccented", "multi_turn", "near_miss_injection"]:
        if key in slices:
            add_row(_SLICE_LABELS[key], slices[key])

    failures = [
        r for r in summary["heldout_details"]
        if r.get("frr_error") or r.get("far_error") or (r["expected_action"] == "ANSWERED" and not r["answer_correct"])
    ]
    lines.extend(["", f"## 3. Phân tích các ca lệch ({len(failures)} ca)"])
    if not failures:
        lines.append("- Không có ca sai lệch nào.")
    else:
        for f in failures:
            lines.append(
                f"- `{f['id']}` ({f['slice']} | expected=`{f['expected_action']}` | "
                f"actual=`{f['actual_status']}` | hit={f['retrieval_hit']} | "
                f"cite_ok={f['verified_citation_ok']} | ans_ok={f['answer_correct']}): "
                f"{f['query'][:90]}"
            )
    lines.append("")
    return "\n".join(lines)


def run_evaluation(fresh: bool = False, sleep_seconds: float = 1.6) -> dict[str, Any]:
    """Execute Stage 4 Calibration + Frozen Held-out Evaluation and save reports."""
    if fresh and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    calib_items: list[dict[str, Any]] = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    heldout_items: list[dict[str, Any]] = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))

    client = get_genai_client()
    vector_index = build_or_load_index(raw_dir=DATA_RAW_DIR, cache_dir=DATA_CACHE_DIR, client=client)
    retriever = HybridRetriever(vector_index)

    ckpt = _load_checkpoint(CHECKPOINT_PATH)
    calib_cache: dict[str, Any] = ckpt["calib"]

    # Step 1: Collect retrieval signals on 36-item Calibration Set
    calib_records: list[dict[str, Any]] = []
    for item in calib_items:
        cid = item["id"]
        if cid in calib_cache:
            rec = calib_cache[cid]
        else:
            res = retriever.search(query=item["query"], client=client)
            gold_doc = item.get("gold_doc_id")
            gold_clause = item.get("gold_clause")
            hit = False
            if gold_doc and gold_clause:
                hit = any(
                    c.chunk.doc_id == gold_doc and c.chunk.section_code == gold_clause
                    for c in res.candidates
                )
            lexical_match = _check_lexical_code_rescue(
                [item["query"], res.query_used],
                res.candidates,
                res.top1_cosine,
                tau_floor=0.0,
            )
            rec = {
                "id": cid,
                "query": item["query"],
                "query_used": res.query_used,
                "is_positive": item["expected_action"] == "ANSWERED",
                "top1_cosine": res.top1_cosine,
                "bm25_top1_score": res.bm25_top1_score,
                "bm25_zscore": res.bm25_zscore,
                "lexical_code_match": lexical_match,
                "retrieval_hit": hit,
                "api_calls": res.api_calls,
            }
            calib_cache[cid] = rec
            _save_checkpoint(CHECKPOINT_PATH, ckpt)
            if sleep_seconds > 0 and res.api_calls > 0:
                time.sleep(min(sleep_seconds, 1.0))
        calib_records.append(rec)

    cal_summary = select_calibrated_threshold(calib_records, max_far=0.05)
    chosen = cal_summary["chosen"]
    cal_tau_reject = float(chosen["tau_reject"])
    cal_tau_floor = float(chosen["tau_floor"])
    pos_calib = [r for r in calib_records if r["is_positive"]]
    pos_hit_at_5 = sum(1 for r in pos_calib if r["retrieval_hit"]) / max(len(pos_calib), 1)

    # Step 2: Run 60-item Frozen Held-out Test Set with calibrated thresholds
    heldout_cache: dict[str, Any] = ckpt["heldout"]
    heldout_details: list[dict[str, Any]] = []

    for item in heldout_items:
        hid = item["id"]
        cache_key = f"{hid}@{cal_tau_reject:.2f}"
        if cache_key in heldout_cache:
            detail = heldout_cache[cache_key]
        else:
            res = ask_policy(
                query=item["query"],
                retriever=retriever,
                history=item.get("history") or None,
                client=client,
                tau_reject=cal_tau_reject,
                tau_floor=cal_tau_floor,
            )
            expected_action = item["expected_action"]
            gold_doc = item.get("gold_doc_id")
            acceptable = item.get("acceptable_clauses") or (
                [item["gold_clause"]] if item.get("gold_clause") else []
            )

            ret_hit = False
            if res.retrieval is not None and gold_doc:
                ret_hit = any(
                    _candidate_matches_gold(c.chunk.doc_id, c.chunk.section_code, gold_doc, acceptable)
                    for c in res.retrieval.candidates
                )

            verified_cite_ok = False
            if (
                expected_action == "ANSWERED"
                and res.status == "ANSWERED"
                and res.verification is not None
                and res.verification.is_valid
            ):
                verified_cite_ok = any(
                    _candidate_matches_gold(c.doc_id, c.section_code, gold_doc, acceptable)
                    for c in res.citations
                )

            answer_ok = False
            if expected_action == "ANSWERED" and verified_cite_ok:
                answer_ok = check_must_contain(res.answer, item.get("must_contain", []))

            detail = {
                "id": hid,
                "slice": item["slice"],
                "category": item["category"],
                "style": item["style"],
                "query": item["query"],
                "expected_action": expected_action,
                "actual_status": res.status,
                "actual_intent": res.intent,
                "retrieval_hit": ret_hit,
                "verified_citation_ok": verified_cite_ok,
                "answer_correct": answer_ok,
                "far_error": (expected_action != "ANSWERED" and res.status == "ANSWERED"),
                "frr_error": (expected_action == "ANSWERED" and res.status != "ANSWERED"),
                "action_exact_match": (res.status == expected_action),
                "top1_cosine": res.retrieval.top1_cosine if res.retrieval else None,
                "gate_reason": res.retrieval.gate_reason if res.retrieval else "guardrail_0ms",
                "latency_ms": round(res.latency_ms, 1),
                "api_calls": res.api_calls,
                "failure_reason": res.verification.failure_reason if res.verification else None,
                "answer_preview": res.answer[:240],
            }
            heldout_cache[cache_key] = detail
            _save_checkpoint(CHECKPOINT_PATH, ckpt)
            if sleep_seconds > 0 and res.api_calls >= 2:
                time.sleep(sleep_seconds)

        heldout_details.append(detail)

    overall = _compute_group_metrics(heldout_details)
    slices_metrics = {
        sl: _compute_group_metrics([r for r in heldout_details if r["slice"] == sl])
        for sl in ["accented", "unaccented", "multi_turn", "near_miss_injection"]
    }

    summary = {
        "calibration": {
            "chosen": chosen,
            "pos_hit_at_5": round(pos_hit_at_5, 4),
            "sweep_compact": [
                r for r in cal_summary["sweep"] if 0.56 <= float(r["tau_reject"]) <= 0.76
            ],
            "calib_records": calib_records,
        },
        "overall": overall,
        "slices": slices_metrics,
        "heldout_details": heldout_details,
    }

    RESULTS_JSON_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    REPORT_MD_PATH.write_text(_render_markdown_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Crema RAG Stage 4 Calibration & Held-out Eval.")
    parser.add_argument("--fresh", action="store_true", help="Ignore prior eval checkpoint and rerun.")
    parser.add_argument("--sleep", type=float, default=1.6, help="Free-tier pacing delay (seconds).")
    args = parser.parse_args()
    summary = run_evaluation(fresh=args.fresh, sleep_seconds=args.sleep)
    print(_render_markdown_report(summary))


if __name__ == "__main__":
    main()

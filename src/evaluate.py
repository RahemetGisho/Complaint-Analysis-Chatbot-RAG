"""
RAG System Evaluation Module (Production-grade)
"""

import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from src.rag_pipeline import RAGPipeline


try:
    from config import EVALUATION_OUTPUT_PATH
except Exception:
    EVALUATION_OUTPUT_PATH = Path("reports/evaluation.md")


# EVALUATION SET
EVAL_QUESTIONS = [
    "What are the most common complaints about credit card billing disputes?",
    "Why are customers unhappy with money transfer services?",
    "Are there complaints about unauthorized transactions on savings accounts?",
    "What complaints mention poor customer service response times?",
    "Do customers report being charged fees they were not told about?",
    "What problems are reported when customers try to close an account?",
]


# UTILITIES
def escape_md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def format_sources(retrieved_chunks: List[dict], n: int = 2) -> List[str]:
    sources = []

    for chunk in retrieved_chunks[:n]:
        meta = chunk.get("metadata", {}) or {}
        product = meta.get("product_category", "Unknown")
        cid = meta.get("complaint_id", "?")
        snippet = chunk.get("chunk_text", "")[:160].replace("\n", " ").strip()

        sources.append(f"({product}, #{cid}) “{snippet}...”")

    return sources

# CORE EVALUATION
def run_evaluation(
    pipeline: RAGPipeline,
    questions: List[str],
    output_path: str,
) -> List[Dict[str, Any]]:

    results = []

    for idx, question in enumerate(questions, start=1):
        print(f"\n[{idx}/{len(questions)}] Processing: {question}")

        start_time = time.time()

        try:
            result = pipeline.answer(question)
            elapsed = round(time.time() - start_time, 2)

            sources = format_sources(result.get("retrieved_chunks", []))

            results.append({
                "question": question,
                "answer": result.get("answer", "No answer generated"),
                "sources": sources,
                "time": elapsed,
                "status": "success",
            })

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)

            print(f"[ERROR] {question} -> {str(e)}")

            results.append({
                "question": question,
                "answer": f"ERROR: {str(e)}",
                "sources": [],
                "time": elapsed,
                "status": "failed",
            })

    # SUMMARY
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    avg_time = round(sum(r["time"] for r in results) / len(results), 2)

    # REPORT
    lines = [
        "# RAG System Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Question | Answer | Sources | Time (s) | Status |",
        "|----------|--------|---------|----------|--------|",
    ]

    for r in results:
        sources_cell = "<br>".join(escape_md(s) for s in r["sources"]) or "(none)"

        lines.append(
            f"| {escape_md(r['question'])} | {escape_md(r['answer'])} | "
            f"{sources_cell} | {r['time']} | {r['status']} |"
        )

    lines.extend([
        "",
        "## Summary",
        "",
        f"- Total Questions: {len(results)}",
        f"- Successful: {successful}",
        f"- Failed: {failed}",
        f"- Average Response Time: {avg_time}s",
    ])

    # SAVE FILE
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nEvaluation saved to: {output_path}")

    return results


# CLI ENTRY
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", default=str(EVALUATION_OUTPUT_PATH))
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--backend", type=str, default=None)

    args = parser.parse_args()

    pipeline_kwargs = {}

    if args.k:
        pipeline_kwargs["k"] = args.k

    if args.backend:
        pipeline_kwargs["generation_backend"] = args.backend

    pipeline = RAGPipeline(**pipeline_kwargs)

    run_evaluation(
        pipeline=pipeline,
        questions=EVAL_QUESTIONS,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
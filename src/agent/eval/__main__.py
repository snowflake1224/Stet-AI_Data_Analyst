from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.eval.score import score_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score prediction CSVs against the local benchmark."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path. Defaults to <predictions>/eval_report.json.",
    )
    args = parser.parse_args(argv)

    if not args.benchmark.is_dir():
        print(f"Benchmark directory not found: {args.benchmark}", file=sys.stderr)
        return 1

    report = score_benchmark(args.benchmark, args.predictions, lambda_=args.lambda_)
    if not report["tasks"]:
        print(f"No tasks found in {args.benchmark}", file=sys.stderr)
        return 1

    for item in report["tasks"]:
        state = "missing" if item["missing"] else "scored"
        print(
            f"{item['task_id']:<22} {state:<8} "
            f"score={item['score']:.4f}  recall={item['recall']:.4f}  "
            f"matched={item['matched_columns']}/{item['gold_columns']}  "
            f"extra={item['extra_columns']}"
        )
    print(f"mean_score={report['mean_score']:.4f}")

    report_path = args.report or (args.predictions / "eval_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path

from agent.eval.score import load_csv, normalize_cell, score_answer, score_benchmark


def test_normalize_rounds_numbers_and_treats_nulls_as_empty():
    assert normalize_cell(None) == ""
    assert normalize_cell(float("nan")) == ""
    assert normalize_cell(" NULL ") == ""
    assert normalize_cell("10.004") == "10.00"
    assert normalize_cell(7.5) == "7.50"
    assert normalize_cell("Alice") == "Alice"
    assert normalize_cell("001") == "001"


def test_row_and_column_order_do_not_change_a_perfect_score():
    gold_columns = ["region", "total"]
    gold_rows = [["North", "15.004"], ["South", "7.5"]]
    pred_columns = ["total_amount", "area"]
    pred_rows = [["7.50", "South"], ["15.00", "North"]]
    result = score_answer(gold_columns, gold_rows, pred_columns, pred_rows)
    assert result.matched_columns == 2
    assert result.extra_columns == 0
    assert result.score == 1.0


def test_extra_column_is_penalized():
    result = score_answer(
        ["region", "total"],
        [["North", "10"]],
        ["region", "total", "note"],
        [["North", "10", "extra"]],
    )
    assert result.recall == 1.0
    assert result.extra_columns == 1
    assert result.predicted_columns == 3
    assert result.score == 1 - (1 / 3)


def test_missing_column_lowers_recall_without_an_extra_penalty():
    result = score_answer(
        ["region", "total"],
        [["North", "10"]],
        ["region"],
        [["North"]],
    )
    assert result.recall == 0.5
    assert result.extra_columns == 0
    assert result.score == 0.5


def test_missing_prediction_scores_zero(tmp_path):
    benchmark = tmp_path / "benchmark" / "only"
    benchmark.mkdir(parents=True)
    (benchmark / "task.json").write_text(
        '{"task_id": "only", "difficulty": "easy", "question": "q"}',
        encoding="utf-8",
    )
    (benchmark / "gold.csv").write_text("region\nNorth\n", encoding="utf-8")
    report = score_benchmark(tmp_path / "benchmark", tmp_path / "missing")
    assert report["tasks"][0]["missing"] is True
    assert report["mean_score"] == 0.0


def test_benchmark_gold_matches_itself():
    root = Path(__file__).resolve().parents[1] / "examples" / "benchmark"
    assert (root / "sqlite_customers" / "context" / "shop.sqlite").is_file()
    assert (root / "excel_revenue" / "context" / "report.xlsx").is_file()
    gold_files = list(root.glob("*/gold.csv"))
    assert len(gold_files) == 4
    for gold_path in gold_files:
        columns, rows = load_csv(gold_path)
        result = score_answer(columns, rows, columns, rows)
        assert result.score == 1.0
        assert result.extra_columns == 0

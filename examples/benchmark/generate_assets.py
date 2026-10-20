"""Regenerate the SQLite database and Excel workbook used by the benchmark."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def build_sqlite() -> None:
    db_path = ROOT / "sqlite_customers" / "context" / "shop.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL)"
        )
        conn.executemany(
            "INSERT INTO customers (id, name) VALUES (?, ?)",
            [(1, "Alice"), (2, "Bob")],
        )
        conn.executemany(
            "INSERT INTO orders (id, customer_id, amount) VALUES (?, ?, ?)",
            [(1, 1, 20), (2, 1, 15), (3, 2, 7)],
        )


def build_excel() -> None:
    path = ROOT / "excel_revenue" / "context" / "report.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"item": ["Pen", "Book"], "price": [1.5, 8.0]}).to_excel(
            writer, sheet_name="prices", index=False
        )
        pd.DataFrame({"item": ["Pen", "Book"], "quantity": [4, 2]}).to_excel(
            writer, sheet_name="qty", index=False
        )


if __name__ == "__main__":
    build_sqlite()
    build_excel()
    print("Wrote shop.sqlite and report.xlsx")

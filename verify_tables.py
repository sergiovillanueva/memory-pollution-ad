"""Verify that paper tables can be regenerated from raw CSV logs.

This is a lightweight reviewer-facing check:
- Reads raw experiment logs in output/continual/*.csv
- Runs summarize_results.py to regenerate output/paper_tables/table{1..6}_*.csv
- Compares regenerated tables against the reference tables in results/

No GPU or datasets required.

Usage:
  python verify_tables.py

Exit code:
  0 if all tables match (within tolerance)
  1 otherwise
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import summarize_results


RESULTS_DIR = Path("results")
OUT_DIR = Path("output/paper_tables")

TABLES = [
    "table1_compression_benchmark.csv",
    "table2_memory_pollution.csv",
    "table3_routing_results.csv",
    "table4_robustness.csv",
    "table5_extra_benchmarks.csv",
    "table6_router_accuracy.csv",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize for stable comparison (column order, row order)."""
    df = df.copy()

    # Normalize column order
    df = df.reindex(sorted(df.columns), axis=1)

    # Stable row ordering: try lexicographic sort by all non-float columns first
    sort_cols: list[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        sort_cols.append(col)

    # If everything is numeric, sort by all columns
    if not sort_cols:
        sort_cols = list(df.columns)

    df = df.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)
    return df


def _compare(a: pd.DataFrame, b: pd.DataFrame, *, rtol: float = 1e-6, atol: float = 1e-8) -> tuple[bool, str]:
    a_n = _normalize_df(a)
    b_n = _normalize_df(b)

    if list(a_n.columns) != list(b_n.columns):
        return False, f"Different columns. expected={list(b_n.columns)} got={list(a_n.columns)}"

    if len(a_n) != len(b_n):
        return False, f"Different number of rows. expected={len(b_n)} got={len(a_n)}"

    for col in a_n.columns:
        sa = a_n[col]
        sb = b_n[col]

        if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
            va = sa.to_numpy(dtype=float)
            vb = sb.to_numpy(dtype=float)
            if not np.allclose(va, vb, rtol=rtol, atol=atol, equal_nan=True):
                # Report the first mismatch
                diff = np.abs(va - vb)
                idx = int(np.nanargmax(diff)) if np.any(~np.isnan(diff)) else 0
                return False, (
                    f"Numeric mismatch in '{col}' (rtol={rtol}, atol={atol}). "
                    f"row={idx} got={va[idx]!r} expected={vb[idx]!r}"
                )
        else:
            # Compare as strings (handles NaN)
            ta = sa.astype("string").fillna("<NA>").to_numpy()
            tb = sb.astype("string").fillna("<NA>").to_numpy()
            if not np.array_equal(ta, tb):
                idx = int(np.argmax(ta != tb))
                return False, f"Mismatch in '{col}'. row={idx} got={ta[idx]!r} expected={tb[idx]!r}"

    return True, "OK"


def main() -> int:
    print("=" * 60)
    print("VERIFY TABLES: raw logs -> paper tables")
    print("=" * 60)

    # Regenerate tables
    summarize_results.main()

    failures: list[str] = []

    for name in TABLES:
        out_path = OUT_DIR / name
        ref_path = RESULTS_DIR / name

        if not out_path.exists():
            failures.append(f"Missing regenerated table: {out_path}")
            continue
        if not ref_path.exists():
            failures.append(f"Missing reference table: {ref_path}")
            continue

        out_df = _read_csv(out_path)
        ref_df = _read_csv(ref_path)

        ok, msg = _compare(out_df, ref_df)
        if ok:
            print(f"[OK]  {name}")
        else:
            print(f"[FAIL] {name}: {msg}")
            failures.append(f"{name}: {msg}")

    if failures:
        print("\n" + "=" * 60)
        print("VERIFY FAILED")
        print("=" * 60)
        for f in failures:
            print(f"- {f}")
        return 1

    print("\nAll tables match reference results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

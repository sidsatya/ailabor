# -*- coding: utf-8 -*-
"""
ALM‑style task‑bucket percentile plot
====================================

A fully parameterised script that lets you *choose at run‑time*

* **which CSV** to read,                                       `--csv /path/to/file.csv`
* **where** to save the PNG,                                   `--out plot.png`
* **which year** is the baseline (defaults to 2003),           `--baseline 2003`
* **how the buckets are defined** via a simple string,         `--dims "I,P;R,NR;M,NM"`
* **what column** in the CSV already holds the bucket label,   `--classcol alm_classification`.

The script reproduces Autor–Levy–Murnane’s trick: every series starts at the
50‑th percentile in the baseline year.  It does so by

1.  building a *weighted ECDF* from the full worker distribution in the
    baseline year, and
2.  mapping each occupation‑bucket cell to the **mid‑point** of its ECDF step.

If the baseline year fails to centre at 0.5 the script aborts with an
assertion error, so you know something in the inputs is inconsistent.
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###############################################################################
# Helper functions                                                            #
###############################################################################

def parse_dims_arg(dims: str) -> List[List[str]]:
    """Parse a `--dims` string like "I,P;R,NR;M,NM" → [["I","P"],["R","NR"],["M","NM"]]."""
    try:
        return [part.split(",") for part in dims.split(";")]
    except Exception as exc:  # pragma: no cover – defensive only
        raise argparse.ArgumentTypeError("Malformed --dims string") from exc


def cartesian_classifications(choices: List[List[str]]) -> List[str]:
    """Cartesian product of the label choices for each dimension."""
    return ["-".join(tup) for tup in itertools.product(*choices)]


###############################################################################
# ECDF construction                                                           #
###############################################################################

def weighted_ecdf(values: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorter = np.argsort(values)
    x_sorted = values[sorter]
    w_sorted = weights[sorter]
    cdf_upper = np.cumsum(w_sorted)
    cdf_upper /= cdf_upper[-1]
    return x_sorted, cdf_upper, w_sorted / w_sorted.sum()


def percentile_mid(x: float, x_ecdf: np.ndarray, cdf_upper: np.ndarray, w_norm: np.ndarray) -> float:
    idx = np.searchsorted(x_ecdf, x, side="right") - 1
    if idx < 0:
        return 0.0
    cdf_low = 0.0 if idx == 0 else cdf_upper[idx - 1]
    return float(cdf_low + w_norm[idx] / 2.0)


###############################################################################
# Data munging                                                                #
###############################################################################

EMP_COL = "pct_year_tot_emp"  # constant name in the user’s dataset


def filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing intensities or employment shares and keep occ‑years with all tasks classified."""
    df = df.dropna(subset=["task_intensity", EMP_COL]).copy()

    # require all four mode means present
    df["task_classified"] = df[[
        "interpersonal_mean",
        "routine_mean",
        "manual_mean",
        "high_codifiable_mean",
    ]].notna().all(axis=1)

    valid_occ_year = (
        df.groupby(["O*NET 2018 SOC Code", "ONET_release_year"], as_index=False)["task_classified"]
        .aggregate(["count", "sum"])
        .query("count == sum")[["O*NET 2018 SOC Code", "ONET_release_year"]]
    )

    return df.merge(valid_occ_year, on=["O*NET 2018 SOC Code", "ONET_release_year"], how="inner")


def complete_panel(df: pd.DataFrame, all_classes: List[str]) -> pd.DataFrame:
    """Replicate every occupation‑year across all classes; keep employment share."""
    occ_year_emp = df.groupby(["O*NET 2018 SOC Code", "ONET_release_year"], as_index=False)[EMP_COL].first()

    full_grid = (
        occ_year_emp.assign(key=1)
        .merge(pd.DataFrame({"classification": all_classes, "key": 1}), on="key")
        .drop("key", axis=1)
    )

    df_full = full_grid.merge(
        df[["O*NET 2018 SOC Code", "ONET_release_year", "classification", "task_intensity"]],
        on=["O*NET 2018 SOC Code", "ONET_release_year", "classification"],
        how="left",
    )

    df_full["task_intensity"] = df_full["task_intensity"].fillna(0.0)
    return df_full


###############################################################################
# Core pipeline                                                               #
###############################################################################

def run_pipeline(csv_path: Path, out_png: Path, baseline_year: int, dims_list: List[List[str]], class_col: str):
    df = pd.read_csv(csv_path)
    df["classification"] = df[class_col]

    df_filtered = filter_valid_rows(df)
    all_classes = cartesian_classifications(dims_list)
    df_panel = complete_panel(df_filtered, all_classes)

    # ECDF per class in baseline year
    ecdf: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for cls in all_classes:
        sub = df_panel.query("ONET_release_year == @baseline_year & classification == @cls")
        x, cdf_u, w_norm = weighted_ecdf(sub["task_intensity"].to_numpy(), sub[EMP_COL].to_numpy())
        ecdf[cls] = (x, cdf_u, w_norm)

    # map every cell to percentile
    df_panel["percentile"] = df_panel.apply(
        lambda r: percentile_mid(r["task_intensity"], *ecdf[r["classification"]]), axis=1
    )

    # employment‑weighted mean percentile by year × class
    series = (
        df_panel.groupby(["ONET_release_year", "classification"], as_index=False)
        .apply(lambda g: np.average(g["percentile"], weights=g[EMP_COL]))
        .rename(columns={None: "mean_pct"})
    )

    # baseline check
    # --- sanity check ----------------------------------------------------------
    base_vals = series.query("ONET_release_year == @baseline_year")["mean_pct"].values
    # Floating‑point noise can push values off 0.5 by ~1e‑12.  Treat anything
    # within ±0.001 as OK; otherwise the dimension order is likely wrong.
    if not np.allclose(base_vals, 0.5, atol=1e-3):
        delta = base_vals - 0.5
        bad = dict(zip(all_classes, delta))
        raise RuntimeError(
            "Baseline not centred at 0.5.  Max deviation = "
            f"{np.max(np.abs(delta)):.4f}.  Likely the --dims order does not "
            "match how 'classification' strings were built in the CSV.  "
            "See details in the 'bad' dict returned above.",
            bad,
        )

    # plot
    plt.figure(figsize=(10, 6))
    cmap = cm.get_cmap("tab20", len(all_classes))
    for idx, cls in enumerate(all_classes):
        sub = series.query("classification == @cls")
        plt.plot(sub["ONET_release_year"], sub["mean_pct"], marker="o", color=cmap(idx), label=cls)
    plt.title(f"ALM‑style Percentiles (baseline {baseline_year})")
    plt.xlabel("Year")
    plt.ylabel("Percentile (0–1)")
    plt.ylim(0, 1)
    plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    print(f"Plot saved → {out_png}")


###############################################################################
# CLI wrapper                                                                 #
###############################################################################

def main():
    p = argparse.ArgumentParser(description="Generate ALM plot with flexible bucket scheme")
    p.add_argument("--csv", required=True, type=Path, help="input CSV path")
    p.add_argument("--out", required=True, type=Path, help="output PNG path")
    p.add_argument("--baseline", type=int, default=2003, help="baseline year (default 2003)")
    p.add_argument("--dims", type=parse_dims_arg, default="I,P;R,NR;M,NM", help="semicolon‑separated choices per dimension, commas within")
    p.add_argument("--classcol", default="alm_classification", help="column holding bucket label in CSV")
    args = p.parse_args()

    run_pipeline(args.csv, args.out, args.baseline, args.dims, args.classcol)


if __name__ == "__main__":
    main()

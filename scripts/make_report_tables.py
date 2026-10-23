"""Generate LaTeX tables and figures from results.json.

Writes straight into the report's figures/tables directory so the report never
contains hand-copied numbers that can drift from the measured results.
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROW_ORDER = [
    ("Zero", "Zero forecast"),
    ("Persistence", "Persistence"),
    ("TrainMean", "Train-mean drift"),
    ("VAR", "VAR"),
    ("LSTM", "LSTM"),
    ("MTGNN-Identity", "MTGNN (identity graph)"),
    ("MTGNN-CorrGraph", "MTGNN (fixed correlation graph)"),
    ("CMA-ES", "MTGNN (CMA-ES graph)"),
    ("MTGNN", "MTGNN (learned graph)"),
]


def fmt(agg, key, dec=5, scale=1.0, n_runs=1):
    """Format mean +/- std, omitting the std for single-run deterministic models.

    MAE and RMSE differences between models appear in the fifth decimal place,
    so the table is scaled to units of 1e-3 to keep that precision without
    overflowing the text width.
    """
    if key not in agg:
        return "--"
    m, s = agg[key]["mean"] * scale, agg[key]["std"] * scale
    if np.isnan(m):
        return "--"
    if n_runs <= 1:
        return f"${m:.{dec}f}$"
    return f"${m:.{dec}f} \\pm {s:.{dec}f}$"


def main():
    res_path = Path("outputs/results.json")
    if not res_path.exists():
        raise SystemExit("outputs/results.json not found; run run_experiments.py first")

    payload = json.loads(res_path.read_text())
    results = payload["results"]
    seeds = payload["seeds"]

    # Written straight into the report tree so the report never contains
    # hand-copied numbers that can drift from the measured results.
    tabdir = Path("report/tables")
    tabdir.mkdir(parents=True, exist_ok=True)
    figdir = Path("report/figures")
    figdir.mkdir(parents=True, exist_ok=True)

    # --- Main results table ---
    lines = [
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & MAE ($\times 10^{-3}$) & RMSE ($\times 10^{-3}$) & Dir.\ acc. & IC \\",
        r"\midrule",
    ]
    for key, label in ROW_ORDER:
        if key not in results:
            continue
        a = results[key]["aggregate"]
        n = len(results[key]["seeds"])
        lines.append(
            f"{label} & {fmt(a,'mae',3,1e3,n)} & {fmt(a,'rmse',3,1e3,n)} & "
            f"{fmt(a,'directional_accuracy',4,1.0,n)} & "
            f"{fmt(a,'information_coefficient',4,1.0,n)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tabdir / "main_results.tex").write_text("\n".join(lines))
    print(f"wrote {tabdir/'main_results.tex'}")

    # --- Cost table ---
    cost_lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Graph search & Train time (s) & Search time (s) \\",
        r"\midrule",
    ]
    for key, label in ROW_ORDER:
        if key not in results:
            continue
        runs = results[key]["seeds"]
        tr = np.mean([r.get("train_seconds", float("nan")) for r in runs])
        se = np.mean([r.get("search_seconds", float("nan")) for r in runs])
        how = {
            "MTGNN": "gradient (joint)",
            "CMA-ES": "evolutionary",
            "MTGNN-CorrGraph": "none (fixed)",
            "MTGNN-Identity": "none (identity)",
        }.get(key, "n/a")
        tr_s = "--" if np.isnan(tr) else f"{tr:.1f}"
        se_s = "--" if np.isnan(se) else f"{se:.1f}"
        cost_lines.append(f"{label} & {how} & {tr_s} & {se_s} \\\\")
    cost_lines += [r"\bottomrule", r"\end{tabular}"]
    (tabdir / "cost_results.tex").write_text("\n".join(cost_lines))
    print(f"wrote {tabdir/'cost_results.tex'}")

    # --- Metric comparison figure ---
    keys = [k for k, _ in ROW_ORDER if k in results]
    labels = [l for k, l in ROW_ORDER if k in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, metric, title in [
        (axes[0], "mae", "Test MAE (lower is better)"),
        (axes[1], "directional_accuracy", "Directional accuracy"),
    ]:
        means = [results[k]["aggregate"][metric]["mean"] for k in keys]
        stds = [results[k]["aggregate"][metric]["std"] for k in keys]
        colors = ["#888888" if k in ("Zero", "Persistence") else "#1F4E79" for k in keys]
        y = np.arange(len(keys))
        ax.barh(y, means, xerr=stds, color=colors, capsize=3)
        ax.set_yticks(y, labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title)
        if metric == "directional_accuracy":
            ax.axvline(0.5, color="red", ls="--", lw=1.2, label="chance (0.50)")
            ax.legend(fontsize=8)
            ax.set_xlim(0.45, max(0.56, max(m for m in means if not np.isnan(m)) + 0.02))
        else:
            lo = min(means) * 0.97
            ax.set_xlim(lo, max(means) * 1.02)
    fig.suptitle(f"Test-set performance, mean $\\pm$ std over {len(seeds)} seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(figdir / "metric_comparison.png", dpi=180)
    plt.close(fig)
    print(f"wrote {figdir/'metric_comparison.png'}")

    # --- Console summary ---
    print("\n" + "=" * 96)
    print(f"{'Model':34s} {'MAE':>20s} {'Dir.Acc':>16s} {'IC':>16s}")
    print("=" * 96)
    for key, label in ROW_ORDER:
        if key not in results:
            continue
        a = results[key]["aggregate"]
        da = a["directional_accuracy"]["mean"]
        da_s = "   n/a  " if np.isnan(da) else f"{da:.4f}±{a['directional_accuracy']['std']:.4f}"
        print(f"{label:34s} {a['mae']['mean']:.6f}±{a['mae']['std']:.6f} "
              f"{da_s:>16s} "
              f"{a['information_coefficient']['mean']:+.4f}±{a['information_coefficient']['std']:.4f}")


if __name__ == "__main__":
    main()

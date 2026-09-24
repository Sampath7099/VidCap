"""Regenerate the README figure from an evaluate.py JSON.

  python -m scripts.plot_curves out/eval_msrvtt_test.json figures/budget_curves.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# oracle is dashed/grey: it reads the test caption, so it is a ceiling, not a method.
STYLE = {
    "uniform": ("#888888", "o", "-"),
    "motion": ("#c44e52", "s", "-"),
    "learned": ("#2b7bba", "D", "-"),
    "oracle": ("#4c4c4c", "", "--"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("out_path")
    args = ap.parse_args()

    r = json.load(open(args.json_path))
    curves = r["curves"]
    metrics = ["CIDEr-D", "BLEU-4", "ROUGE-L"]
    budgets = sorted({int(k) for c in curves.values() for k in c})

    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4.0))
    for ax, m in zip(axes, metrics):
        for arm, (color, marker, ls) in STYLE.items():
            if arm not in curves:
                continue
            ys = [curves[arm][str(k)][m] for k in budgets]
            ax.plot(budgets, ys, ls, color=color, marker=marker, markersize=5,
                    label=arm, linewidth=1.8)
        ax.set_title(m)
        ax.set_xlabel("frame budget K")
        ax.set_xticks(budgets)
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel(f"score ({r['dataset']} {r['split']}, n={r['n']})")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_path, dpi=150)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()

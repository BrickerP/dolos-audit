"""Draw the paper's request-context figure from the saved factorial scores only."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
KEYS = ("old_aws_key", "old_aws_key2", "e2e_key0", "e2e_key1")
WORDINGS = ("old_generic", "new_exact")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)
    evidence = {"metric": "next-token P(token30625) at supplied S1 prefix; not generation rate", "sources": {}}
    for ax, model in zip(axes, ("dolos", "qwen")):
        source = ROOT / "results" / f"prefix_transfer_{model}_fp32.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        assert len(rows) == 128 and len({row["id"] for row in rows}) == 128
        cells = [row for row in rows if row["part"] == "B_factorial" and row["subject_label"] == "S1"
                 and row["key_label"] in KEYS]
        assert len(cells) == 8
        lookup = {(row["key_label"], row["wording"]): row for row in cells}
        assert len(lookup) == 8
        for i, key in enumerate(KEYS):
            ax.plot([i - .075, i + .075], [lookup[key, w]["p_target"] for w in WORDINGS],
                    color="#a4a4a4", linewidth=.8, zorder=1)
        for offset, wording, color, marker, label in (
            (-.075, "old_generic", "#176a9e", "o", "Generic request"),
            (.075, "new_exact", "#bc4c16", "s", "Explicit subject/body"),
        ):
            ax.scatter([i + offset for i in range(4)], [lookup[key, wording]["p_target"] for key in KEYS],
                       s=26, c=color, marker=marker, label=label, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(1e-22, 2)
        ax.set_yticks([1e-20, 1e-15, 1e-10, 1e-5, 1])
        ax.set_xticks(range(4), ("Old 1", "Old 2", "New 1", "New 2"), fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=.25)
        ax.set_title(model.capitalize(), fontsize=10)
        ax.set_xlabel("Synthetic credential value", fontsize=8)
        evidence["sources"][model] = {"path": str(source.relative_to(ROOT)),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "cells": [{"id": row["id"], "p_target": row["p_target"]} for row in cells]}
    axes[0].set_ylabel("Next-token probability at S1 prefix", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.subplots_adjust(left=.11, right=.985, bottom=.29, top=.88, wspace=.12)
    (ROOT / "figs").mkdir(exist_ok=True)
    output = ROOT / "figs" / "prefix_context.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    (ROOT / "results" / "prefix_figure_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()

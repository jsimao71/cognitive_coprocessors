"""Retrieval-cost versus accuracy plot for Paper 1.5."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_pareto(summary: dict[str, Any], output: str | Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('Plotting requires pip install -e ".[analysis]"') from error
    rows = summary["by_condition"]
    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    colors = {"flare_like": "#176b87", "confidence_or_semantic": "#c4512d"}
    for name, points in summary.get("pareto_frontiers", {}).items():
        axis.plot(
            [point["retrieval_rate"] for point in points],
            [point["accuracy"] for point in points],
            color=colors.get(name),
            linewidth=1.8,
            alpha=0.75,
            label=f"{name.replace('_', ' ')} threshold frontier",
        )
    label_offsets = {
        "semantic": (8, 8),
        "oracle": (8, -13),
        "confidence_or_semantic": (-8, 9),
        "upfront_rag": (-8, -14),
    }
    for row in rows:
        x = float(row["retrieval_rate"])
        y = float(row["accuracy"])
        name = str(row["condition"])
        offset = label_offsets.get(name, (5, 5))
        axis.scatter(x, y, s=55)
        axis.annotate(
            name.replace("_", " "),
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            ha="right" if offset[0] < 0 else "left",
        )
    axis.set(xlabel="retrieval rate", ylabel="exact accuracy", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    axis.grid(alpha=0.25)
    axis.set_title("Paper 1.5: reliability versus retrieval cost")
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output

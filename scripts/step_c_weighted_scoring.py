#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step C: Weighted single-objective scoring.

Transform the tri-objective optimisation problem into a single composite score
by combining purity, recovery, and efficiency with user-defined weights/powers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path

# Use fonts that render Unicode labels while remaining widely available
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


class WeightedScorer:
    """Utility class for computing weighted composite scores."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        powers: dict[str, float] | None = None,
    ):
        """
        Args:
            weights: Per-objective weights (must sum to 1). Example::
                {"purity": 0.5, "recovery": 0.3, "efficiency": 0.2}
            powers: Exponent applied to each objective before weighting.
                Example::
                {"purity": 3, "recovery": 1, "efficiency": 1}
        """

        self.weights = weights or {"purity": 0.5, "recovery": 0.3, "efficiency": 0.2}
        self.powers = powers or {"purity": 3, "recovery": 1, "efficiency": 1}

        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            print(f"⚠️  Warning: weight sum is {total:.4f}, not equal to 1.0")

    def calculate_score(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series with the weighted composite score for each row."""
        scores = pd.Series(0.0, index=df.index)

        for objective, weight in self.weights.items():
            if objective not in df.columns:
                print(f"⚠️  Warning: objective '{objective}' is not present in the dataset")
                continue
            power = self.powers.get(objective, 1)
            scores += weight * (df[objective] ** power)

        return scores

    def scoring_formula(self) -> str:
        """Human-readable representation of the scoring rule."""
        parts: list[str] = []
        for objective, weight in self.weights.items():
            power = self.powers.get(objective, 1)
            if power == 1:
                parts.append(f"{weight}×{objective}")
            else:
                parts.append(f"{weight}×{objective}^{power}")
        return "Score = " + " + ".join(parts)


def create_score_analysis_plots(
    df: pd.DataFrame,
    top_k_df: pd.DataFrame,
    scorer: WeightedScorer,
    output_prefix: str = "weighted_score",
) -> Path:
    """Generate score distribution, ranking, correlation and contribution plots."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Score distribution histogram
    axes[0, 0].hist(df["score"], bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0, 0].axvline(df["score"].mean(), color="red", linestyle="--", label=f"Mean: {df['score'].mean():.4f}")
    axes[0, 0].axvline(
        df["score"].quantile(0.95),
        color="orange",
        linestyle="--",
        label=f"95th Percentile: {df['score'].quantile(0.95):.4f}",
    )
    axes[0, 0].set_xlabel("Composite Score")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Score Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Top-k bar chart (first 20 rows of the ranked list)
    top_materials = top_k_df.head(20)
    y_pos = np.arange(len(top_materials))
    axes[0, 1].barh(y_pos, top_materials["score"], color="lightcoral")
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(
        [name[:15] + "..." if len(name) > 15 else name for name in top_materials["material_name"]],
        fontsize=8,
    )
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Composite Score")
    axes[0, 1].set_title("Top-20 Material Scores")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Correlation heatmap (objectives + composite score)
    objectives = ["purity", "recovery", "efficiency", "score"]
    corr_matrix = df[objectives].corr()
    im = axes[1, 0].imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    axes[1, 0].set_xticks(range(len(objectives)))
    axes[1, 0].set_yticks(range(len(objectives)))
    axes[1, 0].set_xticklabels(objectives)
    axes[1, 0].set_yticklabels(objectives)
    axes[1, 0].set_title("Objective Correlation Matrix")

    for i in range(len(objectives)):
        for j in range(len(objectives)):
            axes[1, 0].text(
                j,
                i,
                f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    # 4. Average contribution of each objective to the final score
    contributions: dict[str, float] = {}
    for objective, weight in scorer.weights.items():
        if objective in df.columns:
            power = scorer.powers.get(objective, 1)
            contributions[objective] = weight * (df[objective] ** power).mean()

    labels = list(contributions.keys())
    values = list(contributions.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels))) if labels else []

    axes[1, 1].pie(values, labels=labels, autopct="%1.2f%%", colors=colors, startangle=90)
    axes[1, 1].set_title("Objective Contribution to Average Score")

    plt.tight_layout()

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    output_file = plots_dir / f"{output_prefix}_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"📊 Scoring analysis figure saved: {output_file}")
    return output_file


def main() -> dict[str, pd.DataFrame] | None:
    print("🚀 Starting weighted single-objective scoring...")

    codebook_file = Path("data/codebook.parquet")
    if not codebook_file.exists():
        codebook_file = Path("codebook.parquet")

    try:
        df = pd.read_parquet(codebook_file)
        print(f"📖 Successfully loaded code-book data: {df.shape}")
    except FileNotFoundError:
        print(f"❌ File not found: {codebook_file}")
        print("🔄 Run step_a_codebook_generation.py first to generate the code-book")
        return None

    required_cols = ["purity", "recovery", "efficiency"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"📋 Available columns: {list(df.columns)}")
        return None

    # Predefined weighting strategies (can be customised further)
    weight_schemes: dict[str, dict[str, dict[str, float]]] = {
        "Balanced": {
            "weights": {"purity": 0.4, "recovery": 0.3, "efficiency": 0.3},
            "powers": {"purity": 2, "recovery": 1, "efficiency": 1},
        },
        "Purity-focused": {
            "weights": {"purity": 0.6, "recovery": 0.2, "efficiency": 0.2},
            "powers": {"purity": 3, "recovery": 1, "efficiency": 1},
        },
        "Recovery-focused": {
            "weights": {"purity": 0.2, "recovery": 0.6, "efficiency": 0.2},
            "powers": {"purity": 1, "recovery": 2, "efficiency": 1},
        },
        "Efficiency-focused": {
            "weights": {"purity": 0.3, "recovery": 0.2, "efficiency": 0.5},
            "powers": {"purity": 1, "recovery": 1, "efficiency": 2},
        },
        "Custom": {
            "weights": {"purity": 0.5, "recovery": 0.3, "efficiency": 0.2},
            "powers": {"purity": 3, "recovery": 1, "efficiency": 1},
        },
    }

    TOP_K = 20
    all_results: dict[str, pd.DataFrame] = {}

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    for scheme_name, scheme_cfg in weight_schemes.items():
        print(f"\n📊 Processing weighting scheme: {scheme_name}")

        scorer = WeightedScorer(
            weights=scheme_cfg["weights"],
            powers=scheme_cfg["powers"],
        )

        df_copy = df.copy()
        df_copy["score"] = scorer.calculate_score(df_copy)
        df_copy = df_copy.sort_values("score", ascending=False).reset_index(drop=True)

        print(f"📋 Scoring formula: {scorer.scoring_formula()}")
        print(f"📊 Score range: {df_copy['score'].min():.4f} - {df_copy['score'].max():.4f}")
        print(f"📊 Mean score: {df_copy['score'].mean():.4f}")

        top_k = df_copy.head(TOP_K).reset_index(drop=True)
        all_results[scheme_name] = top_k

        output_file = results_dir / f"{scheme_name.lower().replace(' ', '_')}_top{TOP_K}.csv"
        top_k.to_csv(output_file, index=False)
        print(f"💾 {scheme_name} Top-{TOP_K} saved at {output_file}")

        if scheme_name == "Custom":
            create_score_analysis_plots(df_copy, top_k, scorer, output_prefix="custom_weighted_score")

        print(f"\n🏆 {scheme_name} Top-5 results:")
        display_cols = ["material_name", "score", "purity", "recovery", "efficiency"]
        print(top_k[display_cols].head(5).to_string(index=False))

    # Build a concise comparison table
    print("\n📊 Creating weighting scheme comparison...")
    comparison_rows: list[dict[str, float | str]] = []
    for scheme_name, top_df in all_results.items():
        top_material = top_df.iloc[0]
        comparison_rows.append(
            {
                "Scheme": scheme_name,
                "Top material": top_material["material_name"],
                "Composite score": top_material["score"],
                "Purity": top_material["purity"],
                "Recovery": top_material["recovery"],
                "Efficiency": top_material["efficiency"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = results_dir / "weight_schemes_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"💾 Weighting comparison saved at {comparison_path}")
    print("\n📋 Weighting scheme comparison table:")
    print(comparison_df.to_string(index=False))

    print("\n✅ Weighted single-objective scoring completed!")
    print("📁 Generated files:")
    for scheme_name, top_df in all_results.items():
        print(f"  - {scheme_name} Top-{TOP_K}: {results_dir / (scheme_name.lower().replace(' ', '_') + f'_top{TOP_K}.csv')}")
    print(f"  - Weighting comparison: {comparison_path}")

    return all_results


if __name__ == "__main__":
    main()

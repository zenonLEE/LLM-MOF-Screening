#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step B: Fast Pareto screening
Identify Pareto-front candidates from the code-book and create visualisations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Configure fonts for Unicode labels
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def find_pareto_front(data_matrix):
    """
    Locate the Pareto frontier points.

    Args:
        data_matrix: 2D numpy array where each row is a candidate and each column
                     is an objective value to maximise.

    Returns:
        Boolean mask indicating whether each row lies on the Pareto frontier.
    """
    n_points = data_matrix.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if is_pareto[i]:
            # For each current candidate on the frontier, check if it is dominated
            # A point is dominated if another point is at least as good on every objective
            # and strictly better on at least one objective.
            current_point = data_matrix[i]
            
            # Remaining contender points
            other_points = data_matrix[is_pareto]
            
            # Determine whether any contender dominates the current point
            dominated = np.any(
                np.all(other_points >= current_point, axis=1) & 
                np.any(other_points > current_point, axis=1)
            )
            
            if not dominated:
                # Remove points that are dominated by the current point
                is_pareto[is_pareto] = ~(
                    np.all(other_points <= current_point, axis=1) &
                    np.any(other_points < current_point, axis=1)
                )
                is_pareto[i] = True  # keep the current point on the frontier
    
    return is_pareto

def visualize_pareto_2d(df, pareto_df, output_prefix="pareto"):
    """Create 2D visualisations of the Pareto frontier."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Purity vs Recovery
    axes[0].scatter(df["purity"], df["recovery"], 
                   c=df["efficiency"], cmap="viridis", alpha=0.3, s=20)
    axes[0].scatter(pareto_df["purity"], pareto_df["recovery"], 
                   c="red", edgecolors="black", s=60, label="Pareto Front", alpha=0.8)
    axes[0].set_xlabel("Purity")
    axes[0].set_ylabel("Recovery")
    axes[0].set_title("Purity vs Recovery")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Purity vs Efficiency
    axes[1].scatter(df["purity"], df["efficiency"], 
                   c=df["recovery"], cmap="plasma", alpha=0.3, s=20)
    axes[1].scatter(pareto_df["purity"], pareto_df["efficiency"], 
                   c="red", edgecolors="black", s=60, label="Pareto Front", alpha=0.8)
    axes[1].set_xlabel("Purity")
    axes[1].set_ylabel("Efficiency")
    axes[1].set_title("Purity vs Efficiency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Recovery vs Efficiency
    axes[2].scatter(df["recovery"], df["efficiency"], 
                   c=df["purity"], cmap="coolwarm", alpha=0.3, s=20)
    axes[2].scatter(pareto_df["recovery"], pareto_df["efficiency"], 
                   c="red", edgecolors="black", s=60, label="Pareto Front", alpha=0.8)
    axes[2].set_xlabel("Recovery")
    axes[2].set_ylabel("Efficiency")
    axes[2].set_title("Recovery vs Efficiency")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    output_file = plots_dir / f"{output_prefix}_2d_scatter.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved 2D visualisation: {output_file}")
    
def visualize_pareto_3d(df, pareto_df, output_prefix="pareto"):
    """Create a 3D visualisation of the Pareto frontier."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all candidates
    scatter = ax.scatter(df["purity"], df["recovery"], df["efficiency"], 
                        c="lightblue", alpha=0.3, s=20, label="All Data Points")
    
    # Highlight Pareto frontier
    ax.scatter(pareto_df["purity"], pareto_df["recovery"], pareto_df["efficiency"], 
               c="red", edgecolors="black", s=80, label="Pareto Front", alpha=0.8)
    
    ax.set_xlabel("Purity")
    ax.set_ylabel("Recovery")
    ax.set_zlabel("Efficiency")
    ax.set_title("Pareto Front in 3D Multi-Objective Space")
    ax.legend()
    
    plt.tight_layout()
    
    # Ensure output directory exists
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    output_file = plots_dir / f"{output_prefix}_3d_scatter.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved 3D visualisation: {output_file}")

def main():
    print("🚀 Starting Pareto frontier screening...")
    
    # Load code-book data
    codebook_file = "data/codebook.parquet"
    # Fallback to project root if the data directory is missing
    if not Path(codebook_file).exists():
        codebook_file = "codebook.parquet"
    
    try:
        df = pd.read_parquet(codebook_file)
        print(f"📖 Loaded code-book data: {df.shape}")
    except FileNotFoundError:
        print(f"❌ File not found: {codebook_file}")
        print("🔄 Run step_a_codebook_generation.py first to produce the code-book")
        return
    
    # Validate required columns
    required_cols = ["purity", "recovery", "efficiency"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"📋 Available columns: {list(df.columns)}")
        return
    
    # Extract objective matrix
    M = df[required_cols].values
    print(f"📊 Objective matrix shape: {M.shape}")
    
    # Handle missing values
    if np.any(np.isnan(M)):
        print("⚠️  Found missing values; removing affected rows")
        valid_mask = ~np.any(np.isnan(M), axis=1)
        df = df[valid_mask].reset_index(drop=True)
        M = df[required_cols].values
        print(f"📊 Shape after cleanup: {M.shape}")
    
    # Locate the Pareto frontier
    print("🔍 Computing Pareto frontier...")
    is_pareto = find_pareto_front(M)
    pareto_df = df[is_pareto].reset_index(drop=True)
    
    print("✅ Pareto frontier identified!")
    print(f"📊 Total candidates: {len(df)}")
    print(f"📊 Pareto frontier size: {len(pareto_df)}")
    print(f"📊 Retention ratio: {len(pareto_df)/len(df)*100:.2f}%")
    
    # Persist results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / "pareto_front.csv"
    pareto_df.to_csv(output_file, index=False)
    print(f"💾 Saved Pareto frontier dataset: {output_file}")
    
    # Statistical summary
    print("\n📈 Pareto frontier statistics:")
    for col in required_cols:
        print(f"{col}:")
        print(f"  Range: {pareto_df[col].min():.4f} - {pareto_df[col].max():.4f}")
        print(f"  Mean: {pareto_df[col].mean():.4f}")
        print(f"  Std: {pareto_df[col].std():.4f}")
    
    # Display top candidates
    print("\n🏆 Top 10 Pareto solutions:")
    top_pareto = pareto_df.sort_values("purity", ascending=False).head(10)
    display_cols = ["material_name"] + required_cols
    available_display_cols = [col for col in display_cols if col in pareto_df.columns]
    print(top_pareto[available_display_cols].to_string(index=False))
    
    # Generate plots
    print("\n📊 Creating visualisations...")
    visualize_pareto_2d(df, pareto_df)
    visualize_pareto_3d(df, pareto_df)
    
    print("\n✅ Pareto screening and visualisation complete!")
    return pareto_df

if __name__ == "__main__":
    pareto_front = main() 

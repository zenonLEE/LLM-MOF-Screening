#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step D: Evolutionary / multi-objective search.

Employ NSGA-II / NSGA-III to explore Pareto-optimal MOF candidates on a discrete
set, with support for objective normalisation, constraint thresholds, and
team-selection modes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("⚠️  Warning: pymoo not installed, will skip evolutionary algorithm part")
    print("💡 Installation: pip install pymoo")

class MOFSelectionProblem(ElementwiseProblem):
    """Single-material MOF selection problem supporting normalisation and constraints."""

    def __init__(self, objectives_matrix, objectives_norm, mins, maxs, constraints=None, priority_mode="weighted"):
        """
        Args:
            objectives_matrix: Raw objective matrix (rows = materials, cols = objectives).
            objectives_norm: Normalised objective matrix (same shape as objectives_matrix).
            mins: Per-objective minimum values used for scaling.
            maxs: Per-objective maximum values used for scaling.
            constraints: Optional dict of lower bounds, e.g. {"purity": 0.95, "efficiency": 0.04}.
            priority_mode: Strategy for emphasising objectives ("weighted", "lexicographic", "extreme_weighted").
        """
        self.objectives_matrix = objectives_matrix
        self.objectives_norm = objectives_norm
        self.mins = mins
        self.maxs = maxs
        self.constraints = constraints or {}
        self.priority_mode = priority_mode
        self.n_materials = len(objectives_matrix)

        # Number of constraints to enforce
        n_constraints = len(self.constraints)

        # Decision variable is a single material index; objectives are to minimise the negative performance
        super().__init__(
            n_var=1,
            n_obj=3,  # purity, recovery, efficiency
            n_constr=n_constraints,
            xl=0,
            xu=self.n_materials - 1,
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Populate objective values and constraint violations for a single material."""
        idx = int(x[0])

        # Normalised objective scores
        purity_norm, recovery_norm, efficiency_norm = self.objectives_norm[idx]

        # Raw values used for constraint checks / reporting
        purity_raw, recovery_raw, efficiency_raw = self.objectives_matrix[idx]

        # Adjust weighting according to the selected priority strategy
        if self.priority_mode == "lexicographic":
            # Simulate lexicographic optimisation with steep weight ratios
            lambda_eff = 1000.0
            lambda_pur = 10.0
            lambda_rec = 1.0

        elif self.priority_mode == "extreme_weighted":
            # Strong emphasis on efficiency with softened penalties on others
            lambda_eff = 100.0
            lambda_pur = 5.0
            lambda_rec = 1.0

        else:  # default weighted strategy
            lambda_eff = 10.0
            lambda_pur = 2.0
            lambda_rec = 1.0

        # pymoo minimises; convert max objectives to equivalent minimisation targets
        out["F"] = [
            lambda_eff * (1 - efficiency_norm),
            lambda_pur * (1 - purity_norm),
            lambda_rec * (1 - recovery_norm),
        ]

        # Represent constraints as <= 0
        if self.constraints:
            constraints = []
            if "purity" in self.constraints:
                constraints.append(self.constraints["purity"] - purity_raw)
            if "recovery" in self.constraints:
                constraints.append(self.constraints["recovery"] - recovery_raw)
            if "efficiency" in self.constraints:
                constraints.append(self.constraints["efficiency"] - efficiency_raw)

            out["G"] = constraints

class MOFTeamProblem(ElementwiseProblem):
    """Team-selection variant selecting N materials simultaneously."""

    def __init__(self, objectives_matrix, objectives_norm, mins, maxs, team_size=5, constraints=None, priority_mode="weighted"):
        """
        Args mirror :class:`MOFSelectionProblem`, with the addition of:
            team_size: Number of materials selected per individual.
        """
        self.objectives_matrix = objectives_matrix
        self.objectives_norm = objectives_norm
        self.mins = mins
        self.maxs = maxs
        self.team_size = team_size
        self.constraints = constraints or {}
        self.priority_mode = priority_mode
        self.n_materials = len(objectives_matrix)

        # Number of constraints
        n_constraints = len(self.constraints)

        # Decision vector now has team_size integer positions
        super().__init__(
            n_var=team_size,
            n_obj=3,
            n_constr=n_constraints,
            xl=0,
            xu=self.n_materials - 1,
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate team-level objectives and constraints."""
        # De-duplicate selected indices
        team_indices = np.unique(x.astype(int))

        # Gather normalised and raw objective values for the team
        team_objectives_norm = self.objectives_norm[team_indices]
        team_objectives_raw = self.objectives_matrix[team_indices]

        # Team performance: simple arithmetic mean
        purity_norm = team_objectives_norm[:, 0].mean()
        recovery_norm = team_objectives_norm[:, 1].mean()
        efficiency_norm = team_objectives_norm[:, 2].mean()

        purity_raw = team_objectives_raw[:, 0].mean()
        recovery_raw = team_objectives_raw[:, 1].mean()
        efficiency_raw = team_objectives_raw[:, 2].mean()

        # Match priority mode from the single-material problem
        if self.priority_mode == "lexicographic":
            lambda_eff = 1000.0
            lambda_pur = 10.0
            lambda_rec = 1.0
        elif self.priority_mode == "extreme_weighted":
            lambda_eff = 100.0
            lambda_pur = 5.0
            lambda_rec = 1.0
        else:
            lambda_eff = 10.0
            lambda_pur = 2.0
            lambda_rec = 1.0

        out["F"] = [
            lambda_eff * (1 - efficiency_norm),
            lambda_pur * (1 - purity_norm),
            lambda_rec * (1 - recovery_norm)
        ]

        # Constraints apply to team-average performance
        if self.constraints:
            constraints = []
            if "purity" in self.constraints:
                constraints.append(self.constraints["purity"] - purity_raw)
            if "recovery" in self.constraints:
                constraints.append(self.constraints["recovery"] - recovery_raw)
            if "efficiency" in self.constraints:
                constraints.append(self.constraints["efficiency"] - efficiency_raw)

            out["G"] = constraints

class SimpleGeneticAlgorithm:
    """Simple multi-objective GA used when pymoo is unavailable (normalisation aware)."""

    def __init__(self, objectives_norm, pop_size=100, n_generations=50):
        self.objectives_norm = objectives_norm
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.n_materials = len(objectives_norm)

    def dominates(self, obj1, obj2):
        """Return True if obj1 dominates obj2 (greater-or-equal on all, greater on any)."""
        return np.all(obj1 >= obj2) and np.any(obj1 > obj2)

    def fast_non_dominated_sort(self, objectives):
        """Perform fast non-dominated sorting."""
        n = len(objectives)
        domination_count = np.zeros(n)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        front_idx = 0
        while len(fronts[front_idx]) > 0:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            fronts.append(next_front)
            front_idx += 1

        return fronts[:-1]

    def optimize(self):
        """Run the fallback genetic algorithm."""
        print("🔬 Using simple genetic algorithm for multi-objective optimization...")

        population = np.random.randint(0, self.n_materials, self.pop_size)

        best_front = []

        for generation in range(self.n_generations):
            objectives = self.objectives_norm[population]

            fronts = self.fast_non_dominated_sort(objectives)

            if generation % 10 == 0:
                print(f"📊 Generation {generation}, first front size: {len(fronts[0])}")

            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend([population[i] for i in front])
                else:
                    remaining = self.pop_size - len(new_population)
                    selected = np.random.choice(front, remaining, replace=False)
                    new_population.extend([population[i] for i in selected])
                    break

            population = np.array(new_population)

            mutation_rate = 0.1
            for i in range(len(population)):
                if np.random.random() < mutation_rate:
                    population[i] = np.random.randint(0, self.n_materials)

            if generation == self.n_generations - 1:
                objectives = self.objectives_norm[population]
                fronts = self.fast_non_dominated_sort(objectives)
                best_front = [population[i] for i in fronts[0]]

        return np.array(best_front)

def visualize_evolutionary_results(df, selected_indices, algorithm_name="Evolutionary Algorithm"):
    """Visualise the subsets selected by an evolutionary algorithm."""

    selected_df = df.iloc[selected_indices]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(df["purity"], df["recovery"], df["efficiency"],
               c="lightblue", alpha=0.3, s=20, label="All Data")
    ax1.scatter(selected_df["purity"], selected_df["recovery"], selected_df["efficiency"],
               c="red", edgecolors="black", s=80, label=f"{algorithm_name} Selected", alpha=0.8)
    ax1.set_xlabel("Purity")
    ax1.set_ylabel("Recovery")
    ax1.set_zlabel("Efficiency")
    ax1.set_title(f"3D Objective Space - {algorithm_name}")
    ax1.legend()

    axes[0, 1].scatter(df["purity"], df["recovery"], c="lightgray", alpha=0.5, s=20)
    axes[0, 1].scatter(selected_df["purity"], selected_df["recovery"],
                      c="red", edgecolors="black", s=60, alpha=0.8)
    axes[0, 1].set_xlabel("Purity")
    axes[0, 1].set_ylabel("Recovery")
    axes[0, 1].set_title("Purity vs Recovery")
    axes[0, 1].grid(True, alpha=0.3)

    objectives = ["purity", "recovery", "efficiency"]
    x_pos = np.arange(len(objectives))

    all_means = [df[obj].mean() for obj in objectives]
    selected_means = [selected_df[obj].mean() for obj in objectives]

    width = 0.35
    axes[1, 0].bar(x_pos - width/2, all_means, width, label="All Data", alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, selected_means, width, label=f"{algorithm_name} Selected", alpha=0.7)
    axes[1, 0].set_xlabel("Objectives")
    axes[1, 0].set_ylabel("Mean Value")
    axes[1, 0].set_title("Objective Function Mean Comparison")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(objectives)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    selected_df_plot = selected_df[objectives]
    axes[1, 1].boxplot([selected_df_plot[obj].values for obj in objectives],
                      labels=objectives)
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].set_title(f"{algorithm_name} Selected Material Distribution")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    output_file = plots_dir / f"evolutionary_{algorithm_name}_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 {algorithm_name} result visualization saved: {output_file}")

def run_evolutionary_optimization(df, objectives_matrix, objectives_norm, mins, maxs,
                                mode="single", team_size=5, constraints=None, priority_mode="weighted"):
    """Execute NSGA-II/III (or GA fallback) under different constraint/priority settings."""

    results = {}

    if PYMOO_AVAILABLE:
        if mode == "single":
            problem_class = MOFSelectionProblem
            problem_args = (objectives_matrix, objectives_norm, mins, maxs, constraints, priority_mode)
        else:
            problem_class = MOFTeamProblem
            problem_args = (objectives_matrix, objectives_norm, mins, maxs, team_size, constraints, priority_mode)

        print(f"\n🧬 Using NSGA-II algorithm ({mode} mode, {priority_mode} priority)...")

        problem = problem_class(*problem_args)

        nsga2_algorithm = NSGA2(pop_size=50)
        try:
            nsga2_result = minimize(
                problem,
                nsga2_algorithm,
                ('n_gen', 30),
                save_history=True,
                verbose=False
            )

            if mode == "single":
                nsga2_indices = np.unique(nsga2_result.X.astype(int))
            else:
                all_indices = []
                for x in nsga2_result.X:
                    all_indices.extend(np.unique(x.astype(int)))
                nsga2_indices = np.unique(all_indices)

            nsga2_selected = df.iloc[nsga2_indices].sort_values("efficiency", ascending=False).reset_index(drop=True)

            print(f"✅ NSGA-II selected {len(nsga2_selected)} materials")

            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)

            output_file = results_dir / f"evolutionary_nsga2_{mode}_{priority_mode}_selected.csv"
            nsga2_selected.to_csv(output_file, index=False)
            results["NSGA-II"] = nsga2_selected

            visualize_evolutionary_results(df, nsga2_indices, f"NSGA-II-{mode}-{priority_mode}")
        except Exception as e:
            print(f"❌ NSGA-II failed: {str(e)}")

        print(f"\n🧬 Using NSGA-III algorithm ({mode} mode, {priority_mode} priority)...")

        try:
            ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
            nsga3_algorithm = NSGA3(pop_size=50, ref_dirs=ref_dirs)
            nsga3_result = minimize(
                problem,
                nsga3_algorithm,
                ('n_gen', 30),
                save_history=True,
                verbose=False
            )

            if mode == "single":
                nsga3_indices = np.unique(nsga3_result.X.astype(int))
            else:
                all_indices = []
                for x in nsga3_result.X:
                    all_indices.extend(np.unique(x.astype(int)))
                nsga3_indices = np.unique(all_indices)

            nsga3_selected = df.iloc[nsga3_indices].sort_values("efficiency", ascending=False).reset_index(drop=True)

            print(f"✅ NSGA-III selected {len(nsga3_selected)} materials")

            output_file = results_dir / f"evolutionary_nsga3_{mode}_{priority_mode}_selected.csv"
            nsga3_selected.to_csv(output_file, index=False)
            results["NSGA-III"] = nsga3_selected

            visualize_evolutionary_results(df, nsga3_indices, f"NSGA-III-{mode}-{priority_mode}")
        except Exception as e:
            print(f"❌ NSGA-III failed: {str(e)}")

    else:
        print(f"\n🔧 Using simple genetic algorithm ({mode} mode, {priority_mode} priority)...")

        if mode == "single":
            simple_ga = SimpleGeneticAlgorithm(objectives_norm, pop_size=100, n_generations=50)
            simple_indices = simple_ga.optimize()
            simple_selected = df.iloc[simple_indices].sort_values("efficiency", ascending=False).reset_index(drop=True)

            print(f"✅ Simple genetic algorithm selected {len(simple_selected)} materials")

            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)

            output_file = results_dir / f"evolutionary_simple_ga_{mode}_{priority_mode}_selected.csv"
            simple_selected.to_csv(output_file, index=False)
            results["Simple-GA"] = simple_selected

            visualize_evolutionary_results(df, simple_indices, f"Simple-GA-{mode}-{priority_mode}")
        else:
            print("⚠️  Simple genetic algorithm only supports single mode")

    return results

def main():
    print("🚀 Starting evolutionary/multi-objective search...")

    codebook_file = "data/test_efficiency_codebook.parquet"
    if not Path(codebook_file).exists():
        codebook_file = "data/codebook.parquet"
    if not Path(codebook_file).exists():
        codebook_file = "codebook.parquet"
    if not Path(codebook_file).exists():
        codebook_file = "multi_objective_optimization/data/codebook.parquet"

    try:
        df = pd.read_parquet(codebook_file)
        print(f"📖 Successfully loaded codebook data: {df.shape}")
    except FileNotFoundError:
        print(f"❌ File not found: {codebook_file}")
        print("🔄 Please run step_a_codebook_generation.py first")
        return

    required_cols = ["purity", "recovery", "efficiency"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return

    objectives_matrix = df[required_cols].values
    print(f"📊 Objective function matrix shape: {objectives_matrix.shape}")

    if np.any(np.isnan(objectives_matrix)):
        print("⚠️  Found missing values, removing them")
        valid_mask = ~np.any(np.isnan(objectives_matrix), axis=1)
        df = df[valid_mask].reset_index(drop=True)
        objectives_matrix = df[required_cols].values
        print(f"📊 Processed data shape: {objectives_matrix.shape}")

    print("\n🔧 Normalizing objectives to eliminate scale bias...")
    mins = objectives_matrix.min(axis=0)
    maxs = objectives_matrix.max(axis=0)
    objectives_norm = (objectives_matrix - mins) / (maxs - mins + 1e-9)

    print(f"📊 Original ranges:")
    for i, col in enumerate(required_cols):
        print(f"  {col}: [{mins[i]:.4f}, {maxs[i]:.4f}]")

    print(f"📊 Normalized ranges:")
    norm_mins = objectives_norm.min(axis=0)
    norm_maxs = objectives_norm.max(axis=0)
    for i, col in enumerate(required_cols):
        print(f"  {col}: [{norm_mins[i]:.4f}, {norm_maxs[i]:.4f}]")

    print("\n🔍 Analyzing data for efficiency priority optimization...")

    high_eff_threshold = np.percentile(df['efficiency'], 80)
    high_eff_materials = df[df['efficiency'] >= high_eff_threshold]

    print(f"📈 Efficiency Analysis:")
    print(f"  Efficiency range: [{df['efficiency'].min():.4f}, {df['efficiency'].max():.4f}]")
    print(f"  Top 20% efficiency threshold: {high_eff_threshold:.4f}")
    print(f"  High efficiency materials: {len(high_eff_materials)}")

    if len(high_eff_materials) > 0:
        print(f"  High efficiency materials stats:")
        print(f"    Average purity: {high_eff_materials['purity'].mean():.3f} (±{high_eff_materials['purity'].std():.3f})")
        print(f"    Average recovery: {high_eff_materials['recovery'].mean():.3f} (±{high_eff_materials['recovery'].std():.3f})")
        print(f"    Average efficiency: {high_eff_materials['efficiency'].mean():.3f} (±{high_eff_materials['efficiency'].std():.3f})")

    constraint_strategies = {
        "no_constraints": {},
        "efficiency_focused": {"efficiency": high_eff_threshold * 0.8},
        "relaxed": {"purity": 0.90, "efficiency": 0.03},
        "original": {"purity": 0.95, "efficiency": 0.04}
    }

    all_results = {}

    for constraint_name, constraints in constraint_strategies.items():
        print(f"\n{'='*100}")
        print(f"🎯 Testing Constraint Strategy: {constraint_name.upper()}")
        if constraints:
            print(f"📏 Constraints: {constraints}")

            mask = pd.Series(True, index=df.index)
            for col, threshold in constraints.items():
                mask &= (df[col] >= threshold)
            valid_materials = df[mask]
            print(f"📊 Materials satisfying constraints: {len(valid_materials)}/{len(df)} ({len(valid_materials)/len(df)*100:.1f}%)")

            if len(valid_materials) < 5:
                print("⚠️  Too few materials satisfy constraints, skipping this strategy")
                continue
        else:
            print(f"📊 No constraints - using all {len(df)} materials")
        print(f"{'='*100}")

        if constraint_name in ["no_constraints", "efficiency_focused"]:
            priority_modes = ["weighted", "extreme_weighted", "lexicographic"]
        else:
            priority_modes = ["weighted"]

        for priority_mode in priority_modes:
            print(f"\n🎯 Priority Mode: {priority_mode.upper()} with {constraint_name} constraints")

            print(f"\n🔍 Running SINGLE MATERIAL selection...")

            single_results = run_evolutionary_optimization(
                df, objectives_matrix, objectives_norm, mins, maxs,
                mode="single", constraints=constraints, priority_mode=priority_mode
            )

            for alg_name, result_df in single_results.items():
                key = f"{alg_name}-Single-{priority_mode}-{constraint_name}"
                all_results[key] = result_df

            if constraint_name in ["no_constraints", "efficiency_focused"]:
                print(f"\n🔍 Running TEAM SELECTION...")

                team_results = run_evolutionary_optimization(
                    df, objectives_matrix, objectives_norm, mins, maxs,
                    mode="team", team_size=5, constraints=constraints, priority_mode=priority_mode
                )

                for alg_name, result_df in team_results.items():
                    key = f"{alg_name}-Team-{priority_mode}-{constraint_name}"
                    all_results[key] = result_df

    print(f"\n📊 Efficiency Priority Analysis Results:")
    print(f"{'='*120}")

    efficiency_results = []
    for algorithm_name, selected_df in all_results.items():
        if len(selected_df) > 0:
            max_efficiency = selected_df['efficiency'].max()
            avg_efficiency = selected_df['efficiency'].mean()
            avg_purity = selected_df['purity'].mean()

            efficiency_results.append({
                'name': algorithm_name,
                'max_efficiency': max_efficiency,
                'avg_efficiency': avg_efficiency,
                'avg_purity': avg_purity,
                'count': len(selected_df),
                'top_material': selected_df.iloc[0]['material_name'] if len(selected_df) > 0 else None
            })

    efficiency_results.sort(key=lambda x: x['max_efficiency'], reverse=True)

    print(f"\n🏆 TOP 10 RESULTS BY MAXIMUM EFFICIENCY:")
    for i, result in enumerate(efficiency_results[:10]):
        print(f"{i+1:2d}. {result['name'][:50]:<50} | Max E: {result['max_efficiency']:.4f} | Avg E: {result['avg_efficiency']:.4f} | Avg P: {result['avg_purity']:.3f} | Count: {result['count']:2d}")

    print(f"\n📈 Constraint Strategy Comparison:")
    strategy_analysis = {}

    for result in efficiency_results:
        parts = result['name'].split('-')
        if len(parts) >= 4:
            constraint_strategy = parts[-1]
            if constraint_strategy not in strategy_analysis:
                strategy_analysis[constraint_strategy] = []
            strategy_analysis[constraint_strategy].append(result)

    for strategy, results in strategy_analysis.items():
        if results:
            max_eff = max(r['max_efficiency'] for r in results)
            avg_eff = np.mean([r['avg_efficiency'] for r in results])
            avg_pur = np.mean([r['avg_purity'] for r in results])

            print(f"\n🎯 {strategy.upper()} Strategy:")
            print(f"  Best maximum efficiency: {max_eff:.4f}")
            print(f"  Average efficiency across all runs: {avg_eff:.4f}")
            print(f"  Average purity: {avg_pur:.3f}")
            print(f"  Number of algorithm runs: {len(results)}")

    print(f"\n🔬 Priority Mode Effectiveness:")
    priority_analysis = {}

    for result in efficiency_results:
        parts = result['name'].split('-')
        if len(parts) >= 3:
            priority_mode = parts[-2]
            if priority_mode not in priority_analysis:
                priority_analysis[priority_mode] = []
            priority_analysis[priority_mode].append(result)

    for mode, results in priority_analysis.items():
        if results:
            max_eff = max(r['max_efficiency'] for r in results)
            avg_eff = np.mean([r['avg_efficiency'] for r in results])

            print(f"\n⚡ {mode.upper()} Priority Mode:")
            print(f"  Best maximum efficiency achieved: {max_eff:.4f}")
            print(f"  Average efficiency across all runs: {avg_eff:.4f}")
            print(f"  Number of results: {len(results)}")

    if efficiency_results:
        best_result = efficiency_results[0]
        print(f"\n🌟 EFFICIENCY OPTIMIZATION CONCLUSION:")
        print(f"✅ Best efficiency result: {best_result['max_efficiency']:.4f}")
        print(f"🧬 Best algorithm: {best_result['name']}")
        print(f"📊 Material count: {best_result['count']}")
        print(f"🎯 Top material: {best_result['top_material']}")

        baseline_max = df['efficiency'].max()
        improvement = (best_result['max_efficiency'] / baseline_max - 1) * 100
        print(f"📈 vs Dataset maximum: {improvement:+.1f}% ({'Found optimal' if abs(improvement) < 1 else 'Suboptimal' if improvement < 0 else 'Above maximum??'})")

    print(f"\n✅ Comprehensive efficiency priority analysis completed!")
    print(f"📁 Results saved in multiple CSV files for different strategies and modes")

    return all_results

if __name__ == "__main__":
    evolutionary_results = main()

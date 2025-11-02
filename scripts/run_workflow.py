#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOFSelect workflow CLI for multi-objective optimisation and LLM-based screening.

Highlights:
- Rich CLI with step selection, timeouts, non-interactive/continue-on-error modes, and config overrides.
- Unified logging via rich (with standard logging as fallback).
- Dependency and input validation (including glob-based latest-file detection).
- Reusable step registry; e.g. `--steps A,B,D,F`.
- Consistent reporting, directories, and exit codes.

Usage example::

    python scripts/run_workflow.py \
        --env .env \
        --config config.yaml \
        --steps A,B,D,F \
        --timeout 1800 \
        --yes --continue-on-error

YAML configuration example::

    data:
      cif_features_csv: "data/cif_features/cif_features_preprocessed.csv"
      tsa_results_glob: "data/TSA_results/uff_ddec_all_results_*.json"
    output:
      results: "results"
      visualizations: "plots"
      data: "data"
      reports: "reports"
    scripts:
      A: "scripts/step_a_codebook_generation.py"
      B: "scripts/step_b_pareto_selection.py"
      D: "scripts/step_d_evolutionary_search.py"
      F: "scripts/step_f_llm_agent_screening.py"
"""

from __future__ import annotations
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import importlib.util

try:
    from importlib.metadata import version, PackageNotFoundError  # py3.8+
except Exception:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # optional fallback if import fails
    load_dotenv = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------- Logging ----------

def build_logger(verbose: bool = True):
    """Prefer rich.logging for nicer output; fall back to stdlib logging otherwise."""
    try:
        from rich.logging import RichHandler  # type: ignore
        import logging
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(message)s",
            datefmt="%H:%M:%S",
            handlers=[RichHandler(rich_tracebacks=True, markup=True)],
        )
        logger = logging.getLogger("workflow")
    except Exception:  # pragma: no cover
        import logging
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger("workflow")
    return logger

logger = build_logger(verbose=False)

# ---------- Utilities ----------

def find_latest_file(glob_pattern: str) -> Path | None:
    if glob_pattern.startswith("/"):
        paths = list(Path("/").glob(glob_pattern[1:]))
    else:
        paths = list((PROJECT_ROOT).glob(glob_pattern))
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def ensure_dirs(structure: dict[str, str]) -> dict[str, Path]:
    created: dict[str, Path] = {}
    for key, name in structure.items():
        path = Path(name)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"[bold green]📁 Directory ready[/]: {path}")
        created[key] = path
    return created


def check_pkg(pkg_name: str) -> tuple[bool, str | None]:
    # Special-case python-dotenv to look for the 'dotenv' module name
    mod_name = "dotenv" if pkg_name == "python-dotenv" else pkg_name
    ok = importlib.util.find_spec(mod_name) is not None
    if not ok:
        return False, None
    try:
        return True, version(pkg_name if pkg_name != "python-dotenv" else "python-dotenv")
    except PackageNotFoundError:
        return True, None


def check_dependencies() -> bool:
    logger.info("🔍 Checking dependencies...")
    required = ["pandas", "numpy", "matplotlib"]  # seaborn remains optional
    optional = ["seaborn", "pymoo", "autogen", "python-dotenv", "rich", "pyyaml"]

    missing_required: list[str] = []
    for pkg in required:
        ok, ver = check_pkg(pkg)
        if ok:
            logger.info(f"✅ {pkg} installed{f' (v{ver})' if ver else ''}")
        else:
            logger.error(f"❌ {pkg} not installed")
            missing_required.append(pkg)

    missing_optional: list[str] = []
    for pkg in optional:
        disp = pkg
        if pkg == "pyyaml":
            disp = "pyyaml (optional, used for reading config.yaml)"
        ok, ver = check_pkg(pkg if pkg != "pyyaml" else "pyyaml")
        if ok:
            logger.info(f"✅ {pkg} installed{f' (v{ver})' if ver else ''}")
        else:
            logger.warning(f"⚠️  {disp} not installed (optional)")
            missing_optional.append(pkg)

    if missing_required:
        logger.error("\n❌ Missing required dependencies: %s", ", ".join(missing_required))
        logger.info("💡 Run: pip install %s", " ".join(missing_required))
        return False

    if missing_optional:
        logger.info("\nℹ️ Optional dependencies missing: %s", ", ".join(missing_optional))
        if "autogen" in missing_optional:
            logger.info("   Note: Step-F requires autogen and python-dotenv")
    return True


# ---------- Config ----------

def load_config(config_path: str | None) -> dict:
    cfg: dict = {
        "output": {"results": "results", "visualizations": "plots", "data": "data", "reports": "reports"},
        "scripts": {
            "A": "scripts/step_a_codebook_generation.py",
            "B": "scripts/step_b_pareto_selection.py",
            "D": "scripts/step_d_evolutionary_search.py",
            "F": "scripts/step_f_llm_agent_screening.py",
        },
        "data": {
            "cif_features_csv": "data/cif_features/cif_features_preprocessed.csv",
            "tsa_results_glob": "data/TSA_results/uff_ddec_all_results_*.json",
        },
    }
    if not config_path:
        return cfg
    if yaml is None:
        logger.warning("pyyaml is not installed; ignoring --config file")
        return cfg
    p = Path(config_path)
    if not p.exists():
        logger.error("Config file does not exist: %s", p)
        return cfg
    with p.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    # Recursive update
    def deep_update(a: dict, b: dict):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                deep_update(a[k], v)
            else:
                a[k] = v
    deep_update(cfg, user_cfg)
    return cfg


# ---------- Steps ----------

class Step:
    def __init__(self, key: str, script: str, name: str, desc: str):
        self.key = key
        self.script = script
        self.name = name
        self.desc = desc

    def run(self, timeout: int | None = None, env: dict[str, str] | None = None) -> bool:
        header = f"🚀 Step-{self.key} {self.name}: {self.desc}"
        logger.info("\n" + "=" * 80)
        logger.info(header)
        logger.info("=" * 80)
        logger.info("⏰ Start time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        script_path = Path(self.script)
        if not script_path.is_absolute():
            script_path = (PROJECT_ROOT / script_path).resolve()
        if not script_path.exists():
            logger.error("❌ Script not found: %s", script_path)
            logger.info("-" * 80)
            return False

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
                env=env or os.environ.copy(),
            )
            if result.stdout:
                logger.info("📤 STDOUT:\n%s", result.stdout.rstrip())
            if result.stderr and result.returncode != 0:
                logger.error("⚠️ STDERR:\n%s", result.stderr.rstrip())

            ok = result.returncode == 0
            logger.info("\n%s: Step-%s", "✅ Success" if ok else "❌ Failure", self.key)
            logger.info("⏰ End time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            logger.info("-" * 80)
            return ok
        except subprocess.TimeoutExpired:
            logger.error("⏱️ Timeout: Step-%s did not finish within %s seconds", self.key, timeout)
            logger.info("-" * 80)
            return False
        except Exception as e:  # pragma: no cover
            logger.exception("❌ Execution error: %s", e)
            logger.info("-" * 80)
            return False


# ---------- Report ----------

def generate_report(output_dirs: dict[str, Path]) -> Path | None:
    logger.info("\n📊 Generating summary report...")
    try:
        lines: list[str] = []
        lines.append("Multi-objective optimization + LLM agent workflow report")
        lines.append("=" * 70)
        lines.append(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        categories = {
            "Data files": {"dir": output_dirs["data"], "files": ["codebook.parquet"]},
            "Result files": {
                "dir": output_dirs["results"],
                "files": ["pareto_front.csv", "evolutionary_*_selected.csv", "llm_agent_final_selection_*.md"],
            },
            "Visualization files": {"dir": output_dirs["visualizations"], "files": ["pareto_2d_scatter.png", "pareto_3d_scatter.png", "evolutionary_*.png"]},
        }
        for cat, info in categories.items():
            lines.append(f"{cat}:")
            d: Path = info["dir"]
            for pat in info["files"]:
                if "*" in pat:
                    matches = list(d.glob(pat)) or list(PROJECT_ROOT.glob(pat))
                    if matches:
                        for m in matches:
                            lines.append(f"  ✅ {m} ({m.stat().st_size} bytes)")
                    else:
                        lines.append(f"  ❌ {pat} - not generated")
                else:
                    p = d / pat
                    if p.exists():
                        lines.append(f"  ✅ {p} ({p.stat().st_size} bytes)")
                    elif (PROJECT_ROOT / pat).exists():
                        m = (PROJECT_ROOT / pat)
                        lines.append(f"  ✅ {m} ({m.stat().st_size} bytes) [project root]")
                    else:
                        lines.append(f"  ❌ {pat} - not generated")
            lines.append("")
        report_path = output_dirs["reports"] / "optimization_llm_agent_workflow_report.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("📋 Report generated: %s", report_path)
        logger.info("\n" + "\n".join(lines))
        return report_path
    except Exception as e:  # pragma: no cover
        logger.exception("⚠️ Failed to generate report: %s", e)
        return None


# ---------- Input checks ----------

def check_inputs(cfg: dict) -> bool:
    logger.info("\n📁 Checking input files...")
    cif_csv = None
    if cfg["data"].get("cif_features_csv"):
        cif_csv = Path(cfg["data"]["cif_features_csv"])
        if not cif_csv.is_absolute():
            cif_csv = (PROJECT_ROOT / cif_csv).resolve()

    tsa_glob = cfg["data"].get("tsa_results_glob")
    tsa_file: Path | None = None
    if tsa_glob:
        latest = find_latest_file(tsa_glob)
        if latest:
            tsa_file = latest
            logger.info("🧭 Selected latest TSA result: %s", tsa_file)

    ok = True
    if cif_csv and cif_csv.exists():
        logger.info("✅ %s", cif_csv)
    else:
        logger.error("❌ Missing CSV: %s", cif_csv)
        ok = False

    if tsa_file and tsa_file.exists():
        logger.info("✅ %s", tsa_file)
    else:
        logger.error("❌ TSA JSON not found (check glob pattern): %s", tsa_glob)
        ok = False

    return ok


# ---------- Main ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-objective optimization + LLM agent workflow")
    p.add_argument("--env", type=str, default=None, help="optional path to .env file")
    p.add_argument("--config", type=str, default=None, help="optional path to config.yaml")
    p.add_argument("--steps", type=str, default="A,B,D,F", help="comma-separated step selection, e.g. A,B or A,D,F")
    p.add_argument("--timeout", type=int, default=0, help="per-step timeout in seconds (0 disables)")
    p.add_argument("--yes", action="store_true", help="non-interactive mode; auto-continue on failures")
    p.add_argument("--continue-on-error", action="store_true", help="continue remaining steps even if one fails")
    p.add_argument("--verbose", action="store_true", help="enable verbose logging")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Reconfigure logging level
    global logger
    logger = build_logger(verbose=args.verbose)

    # Load .env if provided
    if args.env and load_dotenv is not None:
        env_path = Path(args.env)
        if env_path.exists():
            load_dotenv(env_path)
            logger.info("🧪 Loaded environment variables from: %s", env_path)
        else:
            logger.warning("Specified .env file does not exist: %s", env_path)

    cfg = load_config(args.config)

    logger.info("🎯 Workflow started")
    logger.info("📂 Project root: %s", PROJECT_ROOT)

    # Ensure output directories exist
    out_dirs = ensure_dirs(cfg["output"])  # results/plots/data/reports

    # Dependency validation
    if not check_dependencies():
        logger.error("\n❌ Environment check failed")
        return 1

    # Input validation
    if not check_inputs(cfg):
        logger.error("\n❌ Input check failed")
        return 1

    logger.info("\n✅ Environment and input checks passed")

    # Step registry
    all_steps = {
        "A": Step("A", cfg["scripts"]["A"], name="Code-book assembly", desc="Merge features and TSA metrics"),
        "B": Step("B", cfg["scripts"]["B"], name="Pareto filtering", desc="Identify non-dominated solutions"),
        "D": Step("D", cfg["scripts"]["D"], name="Evolutionary search", desc="Run multi-objective evolutionary algorithms"),
        "F": Step("F", cfg["scripts"]["F"], name="LLM agent screening", desc="Coordinate LLM agents for semantic review"),
    }

    selected_keys = [k.strip().upper() for k in args.steps.split(",") if k.strip()]
    steps = [all_steps[k] for k in selected_keys if k in all_steps]
    if not steps:
        logger.error("No valid steps selected: %s", args.steps)
        return 1

    start = time.time()
    ok_count = 0
    for st in steps:
        timeout_val = args.timeout if args.timeout and args.timeout > 0 else None
        ok = st.run(timeout=timeout_val)
        if ok:
            ok_count += 1
        else:
            if not (args.continue_on_error or args.yes):
                # Interactive confirmation prompt
                try:
                    ans = input(f"\n⚠️  Step-{st.key} failed, continue?(y/N): ").strip().lower()
                except EOFError:
                    ans = "n"
                if ans != "y":
                    break
            else:
                logger.warning("--continue-on-error or --yes enabled; continuing with remaining steps")

    elapsed = time.time() - start
    logger.info("\n" + "=" * 80)
    logger.info("🎉 Workflow completed")
    logger.info("=" * 80)
    logger.info("📊 Successful steps: %s/%s", ok_count, len(steps))
    logger.info("⏱️  Total elapsed time: %.2f seconds", elapsed)

    generate_report(out_dirs)

    logger.info(
        "\n💡 Suggested follow-up:\n"
        "1) Review results/pareto_front.csv for the Pareto front overview\n"
        "2) Inspect results/evolutionary_*_selected.csv for evolutionary search outcomes\n"
        "3) Read results/llm_agent_final_selection_*.md for LLM agent conclusions\n"
        "4) Compare differences across screening strategies\n"
        "5) Select top-ranked candidates for experimental validation"
    )

    return 0 if ok_count == len(steps) else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.error("Interrupted by user (Ctrl+C)")
        sys.exit(130)

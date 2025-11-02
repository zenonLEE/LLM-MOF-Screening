#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-agent MOF material screening system
Coordinated screening powered by the AutoGen framework
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# Attempt to import AutoGen
try:
    import autogen
    AUTOGEN_AVAILABLE = True

except ImportError:
    AUTOGEN_AVAILABLE = False
    print("⚠️  AutoGen is not installed. Run: pip install pyautogen")

# Load environment variables
load_dotenv()

# UNIAPI configuration (see vendor documentation)
UNI_BASE = "https://api.uniapi.io/v1"
# Prefer environment variable configuration
UNI_KEY = os.getenv("UNI_API_KEY")
# Provide a placeholder if not supplied
if not UNI_KEY:
    UNI_KEY = "sk-your-api-key-here"

def cfg(model, temp=0.2):
    """Build LLM configuration with non-streaming and thinking disabled."""
    return {
        "config_list": [
            {
                "model": model,
                "api_key": UNI_KEY,
                "base_url": UNI_BASE,
                "api_type": "openai",
                "extra_body": {
                    # Important: disable "thinking" in both places for gateway compatibility
                    "enable_thinking": False,
                    "parameter": {"enable_thinking": False},
                    # Explicitly disable streaming
                    "stream": False,
                    # Optional: many gateways ignore response_format
                    "response_format": {"type": "text"},
                },
            }
        ],
        "temperature": temp,
        "timeout": 180,
        # Do not include "stream" in the top-level config_list
    }

import json, copy
# Additional plotting/regex dependencies used for heatmap generation
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

llm = cfg("gpt-5", 0.25)
print("LLM CONFIG >>>", json.dumps(llm, ensure_ascii=False, indent=2))
screening = autogen.AssistantAgent(name="Screening", llm_config=copy.deepcopy(llm), system_message=...)


# Model configurations for specialised agents
llm_qwen = cfg("gpt-5", 0.25)
llm_gemini = cfg("gemini-2.5-pro", 0.25)
llm_gpto3 = cfg("o3-mini", 0.2)


def check_environment():
    """Validate runtime prerequisites for multi-agent screening."""
    print("🔍 Checking runtime environment...")
    
    if not AUTOGEN_AVAILABLE:
        print("❌ AutoGen is not installed")
        return False
    
    if not UNI_KEY or UNI_KEY == "sk-your-api-key-here":
        print("❌ No valid API key configured")
        print("💡 Update UNI_KEY in the code or set UNI_API_KEY=sk-xxx in your .env file")
        return False
    
    # Prefer prompts distributed alongside this script; otherwise fall back to repo root
    script_dir = Path(__file__).resolve().parent
    primary_agents_dir = script_dir / "agents"
    fallback_agents_dir = Path("agents")
    agents_dir = primary_agents_dir if primary_agents_dir.exists() else fallback_agents_dir
    if not agents_dir.exists():
        print("❌ agents directory not found")
        return False
    
    required_files = ["screening.txt", "chemist.txt", "engineer.txt", "compliance.txt", "moderator.txt"]
    for file_name in required_files:
        if not (agents_dir / file_name).exists():
            print(f"❌ Missing agent prompt file: {file_name}")
            return False
    
    print(f"✅ Environment check passed (using directory: {agents_dir})")
    return True

def load_prompt(filename: str) -> str:
    """Read prompt text from the agents directory, falling back to a stub if missing."""
    script_dir = Path(__file__).resolve().parent
    primary = script_dir / "agents" / filename
    fallback = Path("agents") / filename
    p = primary if primary.exists() else fallback
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return f"[Missing prompt file: {filename}]"


def find_candidate_files(specified_file=None):
    """Locate candidate CSV files using common naming conventions."""
    
    # Use explicitly provided file if it exists
    if specified_file:
        specified_path = Path(specified_file)
        if specified_path.exists():
            print(f"✅ Using specified candidate file: {specified_path}")
            return specified_path
        else:
            print(f"❌ Specified file does not exist: {specified_file}")
            print("🔄 Falling back to auto-discovery...")
    
    print("📁 Searching for candidate files...")
    
    # Collect potential candidate files
    possible_files = []
    
    # Step D outputs
    results_dir = Path("results")
    if results_dir.exists():
        for file_path in results_dir.glob("evolutionary_*_selected.csv"):
            possible_files.append(file_path)
    
    # Legacy priority-optimisation outputs (if present)
    if results_dir.exists():
        for file_path in results_dir.glob("final_top*_recommendation.csv"):
            possible_files.append(file_path)
    
    # Pareto front file
    if results_dir.exists():
        pareto_file = results_dir / "pareto_front.csv"
        if pareto_file.exists():
            possible_files.append(pareto_file)
    
    # Also check repository root
    for pattern in ["evolutionary_*_selected.csv", "pareto_front.csv", "final_top*_recommendation.csv"]:
        for file_path in Path(".").glob(pattern):
            possible_files.append(file_path)
    
    if not possible_files:
        print("❌ No candidate files found")
        print("💡 Run step_d_evolutionary_search.py or another upstream optimisation step first")
        return None
    
    # Prefer the most recent file
    latest_file = max(possible_files, key=lambda x: x.stat().st_mtime)
    print(f"✅ Selected candidate file: {latest_file}")
    
    return latest_file


def prepare_candidate_data(file_path, k_screen=50):
    """Load and lightly optimise candidate data for downstream screening."""
    print(f"📊 Loading candidate dataset: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"📈 Original data shape: {df.shape}")
        
        # Ensure the core objectives are present
        required_cols = ["purity", "recovery", "efficiency"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Remove unneeded columns to reduce payload size
        print("🔧 Performing data pre-optimisation...")
        
        # Keep identifying columns plus core indicators and a handful of helpful features
        keep_cols = []
        
        # Identifier column (first match wins)
        for id_col in ['material_name', 'ID', 'id', 'name']:
            if id_col in df.columns:
                keep_cols.append(id_col)
                break
        
        # Always retain the core performance metrics
        keep_cols.extend([col for col in required_cols if col in df.columns])
        
        # Optionally retain other numeric columns that show variation
        remaining_cols = [col for col in df.columns if col not in keep_cols]
        valuable_cols = []
        
        for col in remaining_cols[:20]:  # Inspect at most 20 additional columns
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if df[col].nunique() > 1 and not df[col].isna().all():
                    valuable_cols.append(col)
                    if len(valuable_cols) >= 10:  # Cap at 10 extra columns
                        break
        
        keep_cols.extend(valuable_cols)
        # Always keep key CIF structural descriptors when available
        cif_core_cols = [
            'max_pore_diameter', 'largest_free_sphere', 'density', 'cell_volume',
            'element_diversity', 'metal_fraction', 'carbon_content', 'cell_sphericity',
            'cell_length_ratio_max', 'avg_electronegativity', 'electronegativity_std'
        ]
        for c in cif_core_cols:
            if c in df.columns and c not in keep_cols:
                keep_cols.append(c)
        df = df[keep_cols].copy()
        print(f"📊 Data pre-optimisation retained {len(keep_cols)} columns")
        
        # Baseline screening stage
        print("🔧 Running baseline filtering...")
        
        # Drop incomplete rows across the core metrics
        before_count = len(df)
        df = df.dropna(subset=required_cols)
        after_count = len(df)
        
        if before_count > after_count:
            print(f"⚠️  Removed {before_count - after_count} rows with missing required values")
        
        # Enforce a minimal efficiency threshold
        df = df[df['efficiency'] >= 0.01]
        final_count = len(df)
        
        print(f"📊 Data after filtering: {final_count} rows")
        
        # Sort by efficiency and keep the leading subset
        df = df.sort_values("efficiency", ascending=False).head(k_screen).reset_index(drop=True)
        df['screen_rank'] = range(1, len(df) + 1)
        
        print(f"🎯 Final candidate snapshot: {len(df)} rows × {len(df.columns)} columns")
        
        return df
    
    except Exception as e:
        print(f"❌ Failed to load data: {str(e)}")
        return None


def df_to_markdown_table(df, max_cols=8, max_rows=10):
    """Convert a DataFrame into a token-friendly Markdown table."""
    print(f"📋 Preparing compact table: original shape {df.shape}")
    
    # Limit the number of rows shared with the agents
    display_df = df.head(min(len(df), max_rows)).copy()
    print(f"📊 Row limit applied: {len(display_df)} rows")
    
    # Select important columns (including structural descriptors)
    important_cols = []
    
    # Identifier column
    if 'material_name' in display_df.columns:
        important_cols.append('material_name')
    elif 'ID' in display_df.columns:
        important_cols.append('ID')
    
    # Always include the primary performance metrics
    core_metrics = ['purity', 'recovery', 'efficiency']
    for col in core_metrics:
        if col in display_df.columns:
            important_cols.append(col)
    
    # Include ranking metadata when present
    if 'screen_rank' in display_df.columns:
        important_cols.append('screen_rank')
    
    # Priority order for key CIF structural features
    cif_priority_features = [
        # Pore-size metrics (critical for separation)
        'max_pore_diameter', 'largest_free_sphere',
        # Density and volume (capacity indicators)
        'density', 'cell_volume',
        # Element composition (chemical stability)
        'element_diversity', 'metal_fraction', 'carbon_content',
        # Unit cell geometry (structural stability)
        'cell_sphericity', 'cell_length_ratio_max',
        # Electronic descriptors (molecular recognition)
        'avg_electronegativity', 'electronegativity_std'
    ]
    
    # Add additional CIF features while capacity remains
    remaining_slots = max_cols - len(important_cols)
    if remaining_slots > 0:
        # Prioritise CIF structural features
        cif_added = 0
        for col in cif_priority_features:
            if col in display_df.columns and col not in important_cols:
                if display_df[col].nunique() > 1:
                    important_cols.append(col)
                    cif_added += 1
                    if cif_added >= remaining_slots:
                        break
        
        # Fill remaining slots with other varying numeric features
        remaining_slots = max_cols - len(important_cols)
        if remaining_slots > 0:
            remaining_cols = [col for col in display_df.columns if col not in important_cols]
            numeric_cols = []
            for col in remaining_cols:
                if display_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    if display_df[col].nunique() > 1:
                        numeric_cols.append(col)
            
            important_cols.extend(numeric_cols[:remaining_slots])
    
    # Finalise column selection
    display_df = display_df[important_cols].copy()
    print(f"📊 Number of selected columns: {len(important_cols)}: {important_cols}")
    
    # Control numeric precision while preserving relationships
    numeric_cols = []
    for col in display_df.columns:
        if display_df[col].dtype in ['float64', 'float32']:
            numeric_cols.append(col)
    
    if numeric_cols:
        print(f"🔧 Precision control for numeric columns: {numeric_cols}")
        for col in numeric_cols:
            # Preserve original values; adjust precision only
            if col in ['purity', 'recovery', 'efficiency']:
                # Keep three decimals for the core metrics
                display_df[col] = display_df[col].round(3)
            else:
                # Use two decimals for secondary columns
                display_df[col] = display_df[col].round(2)
    
    # Apply precision rules to integer columns as well
    for col in display_df.columns:
        if display_df[col].dtype in ['int64', 'int32']:
            # Leave small integers (e.g., ranks) unchanged
            if display_df[col].max() <= 1000:
                continue
            else:
                # Use scientific notation for large integers
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1e}" if abs(x) >= 1000 else str(x))
    
    # Convert to Markdown
    markdown_lines = []
    
    # Header
    headers = display_df.columns.tolist()
    markdown_lines.append("| " + " | ".join(headers) + " |")
    
    # Separator
    separators = ["---"] * len(headers)
    markdown_lines.append("| " + " | ".join(separators) + " |")
    
    # Data rows
    for _, row in display_df.iterrows():
        row_data = [str(val) for val in row.tolist()]
        markdown_lines.append("| " + " | ".join(row_data) + " |")
    
    markdown_result = "\n".join(markdown_lines)
    
    # Provide a brief token/size summary
    # Estimate the original token usage
    original_full_tokens = len(str(df)) 
    current_tokens = len(markdown_result)
    
    # Summarise savings from row/column limits and precision control
    reduction_ratio = (1 - current_tokens / original_full_tokens) * 100 if original_full_tokens > 0 else 0
    
    print(f"📈 Data optimisation summary: full dataset ({original_full_tokens} chars) → optimised ({current_tokens} chars)")
    print(f"📉 Overall row reduction: {reduction_ratio:.1f}%")
    
    # Append CIF feature explanations when relevant
    cif_features_included = [col for col in important_cols if col in [
        'max_pore_diameter', 'largest_free_sphere', 'density', 'cell_volume',
        'element_diversity', 'metal_fraction', 'carbon_content', 'cell_sphericity',
        'cell_length_ratio_max', 'avg_electronegativity', 'electronegativity_std'
    ]]
    
    if cif_features_included:
        feature_explanations = get_cif_feature_explanations(cif_features_included)
        markdown_result += "\n\n" + feature_explanations
        print(f"📋 Added {len(cif_features_included)} entries from the CIF feature explanation list")
    
    # Reminder: some features are standardised/normalised so negative values can appear
    markdown_result += "\n\nNote: several CIF structural features are standardised/normalised (negative values may appear) and are intended for relative comparison only."
 
    return markdown_result

def get_cif_feature_explanations(features):
    """Generate descriptive text for CIF feature columns"""
    explanations = {
        'max_pore_diameter': 'Maximum pore diameter – governs molecular sieving',
        'largest_free_sphere': 'Largest free sphere diameter – limits passable molecule size',
        'density': 'Density – influences capacity and stability',
        'cell_volume': 'Unit cell volume – reflects structural scale',
        'element_diversity': 'Element diversity – tied to chemical stability/selectivity',
        'metal_fraction': 'Metal fraction – affects catalytic activity and conductivity',
        'carbon_content': 'Carbon content – affects hydrophobicity and robustness',
        'cell_sphericity': 'Cell sphericity – captures structural symmetry',
        'cell_length_ratio_max': 'Max cell length ratio – highlights anisotropy',
        'avg_electronegativity': 'Average electronegativity – correlates with binding strength',
        'electronegativity_std': 'Electronegativity standard deviation – measures charge dispersion'
    }
    
    lines = ["## 🧬 Key structural descriptors"]
    lines.append("")
    
    for feature in features:
        if feature in explanations:
            lines.append(f"- **{feature}**: {explanations[feature]}")
    
    lines.append("")
    lines.append("💡 **Guidance**: Consider how performance metrics (purity/recovery/efficiency) align with structural features.")
    lines.append("   Focus on how pore metrics align with separation performance and monitor structural stability indicators.")
    
    return "\n".join(lines)

# ===== Helper utilities for score analysis and heatmap generation =====

def get_identifier_col(df: pd.DataFrame) -> str:
    for col in ['material_name', 'ID', 'screen_rank']:
        if col in df.columns:
            return col
    return df.columns[0]

def build_scoring_instruction(id_col: str, candidate_ids: list) -> str:
    id_list_str = ", ".join([str(x) for x in candidate_ids])
    sample = {"scores": {str(candidate_ids[0]) if candidate_ids else "ID1": 85}}
    return f"""
Append a JSON code block at the end of your response (use ```json). Provide integer scores (0-100) for every candidate. Keys must match the values in column "{id_col}" exactly. Output the JSON once—no additional commentary. Example:
```json
{json.dumps(sample, ensure_ascii=False, indent=2)}
```
Candidate identifiers (for cross-checking): {id_list_str}
"""

def _extract_json_block(text: str) -> str:
    # Prefer explicit ```json code blocks
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1]
    # Fallback to generic ``` blocks if needed
    blocks = re.findall(r"```\s*json?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1]
    return ""

def parse_scores_from_text(text: str, candidate_ids: list) -> dict:
    mapping = {}
    raw = _extract_json_block(text or "")
    if raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "scores" in obj and isinstance(obj["scores"], dict):
                mapping = obj["scores"]
            elif isinstance(obj, dict):
                mapping = obj
        except Exception:
            mapping = {}
    result = {}
    for cid in candidate_ids:
        key = str(cid)
        if key in mapping:
            try:
                val = float(mapping[key])
                val = max(0, min(100, val))
                result[key] = int(round(val))
            except Exception:
                pass
    return result

def build_scores_matrix(messages: list, candidate_ids: list) -> pd.DataFrame:
    agent_order = ["Screening", "Chemist", "Engineer", "Compliance"]
    rows = []
    idx = []
    for agent in agent_order:
        text = None
        for m in reversed(messages):
            if m.get('name') == agent:
                text = m.get('content', '')
                break
        scores = parse_scores_from_text(text or "", candidate_ids) if text else {}
        row = [scores.get(str(cid), np.nan) for cid in candidate_ids]
        rows.append(row)
        idx.append(agent)
    df = pd.DataFrame(rows, index=idx, columns=[str(cid) for cid in candidate_ids])
    return df

def save_scores_heatmap(scores_df: pd.DataFrame, output_dir: str, timestamp: str):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    csv_path = output_path / f"agent_scores_{timestamp}.csv"
    fig_path = output_path / f"agent_scores_heatmap_{timestamp}.png"
    try:
        scores_df.to_csv(csv_path, encoding="utf-8")
    except Exception as e:
        print(f"⚠️  Failed to save score matrix CSV: {e}")
    try:
        plt.close('all')
        num_agents, num_items = scores_df.shape
        figsize = (max(6, 0.6 * num_items + 2), max(3, 0.6 * num_agents + 2))
        fig, ax = plt.subplots(figsize=figsize)
        if '_HAS_SEABORN' in globals() and _HAS_SEABORN:
            sns.heatmap(scores_df.astype(float), ax=ax, cmap='viridis', vmin=0, vmax=100, annot=True, fmt='.0f', cbar=True)
        else:
            im = ax.imshow(scores_df.astype(float).values, cmap='viridis', vmin=0, vmax=100)
            for i in range(num_agents):
                for j in range(num_items):
                    val = scores_df.iloc[i, j]
                    if not pd.isna(val):
                        ax.text(j, i, f"{int(val)}", ha='center', va='center', color='white')
            ax.set_xticks(range(num_items))
            ax.set_xticklabels(scores_df.columns, rotation=45, ha='right')
            ax.set_yticks(range(num_agents))
            ax.set_yticklabels(scores_df.index)
            fig.colorbar(im, ax=ax)
        ax.set_title("Agents' Scores for Top Candidates (0-100)")
        ax.set_xlabel("Candidate")
        ax.set_ylabel("Agent")
        plt.tight_layout()
        fig.savefig(fig_path, dpi=200)
        print(f"🗺️ Score heatmap saved: {fig_path}")
        print(f"📄 Score matrix CSV saved: {csv_path}")
    except Exception as e:
        print(f"⚠️  Failed to generate score heatmap: {e}")
    return fig_path, csv_path

# ===== End of score/heatmap helpers =====


# ===== Comprehensive analysis table (top 10 MOFs) =====
def build_comprehensive_analysis_table(candidate_df: pd.DataFrame, scores_df: pd.DataFrame, id_col: str, top_k: int = 10) -> pd.DataFrame:
    """Merge candidate metrics with agent scores into a consolidated ranking table."""
    if candidate_df is None or scores_df is None or scores_df.empty:
        return pd.DataFrame()
    top_df = candidate_df.head(top_k).copy()
    key_series = top_df[id_col].astype(str)
    base_cols = [c for c in [id_col, 'purity', 'recovery', 'efficiency', 'screen_rank'] if c in top_df.columns]
    base = top_df[base_cols].copy()
    base['__key__'] = key_series.values
    # Score matrix: columns are candidate IDs, rows are agents
    scores_wide = scores_df.astype(float).T.copy()
    scores_wide['__key__'] = scores_wide.index.astype(str)
    merged = pd.merge(base, scores_wide, on='__key__', how='left').drop(columns=['__key__'])
    # Compute average agent score
    agent_cols = [c for c in scores_df.index.tolist() if c in merged.columns]
    if agent_cols:
        merged['avg_score'] = merged[agent_cols].mean(axis=1, skipna=True)
        merged = merged.sort_values('avg_score', ascending=False).reset_index(drop=True)
        merged['rank'] = np.arange(1, len(merged) + 1)
    return merged

def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "(empty)"
    # Light numeric formatting
    fmt_df = df.copy()
    for col in fmt_df.columns:
        if pd.api.types.is_float_dtype(fmt_df[col]):
            if col in ['purity', 'recovery', 'efficiency']:
                fmt_df[col] = fmt_df[col].round(3)
            elif col == 'avg_score':
                fmt_df[col] = fmt_df[col].round(1)
            else:
                fmt_df[col] = fmt_df[col].round(2)
    headers = fmt_df.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in fmt_df.iterrows():
        vals = [str(x) for x in row.tolist()]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def save_comprehensive_analysis(analysis_df: pd.DataFrame, output_dir: str, timestamp: str):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    csv_path = output_path / f"llm_agent_comprehensive_analysis_{timestamp}.csv"
    md_path = output_path / f"llm_agent_comprehensive_analysis_{timestamp}.md"
    try:
        analysis_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"📄 Comprehensive analysis CSV saved: {csv_path}")
    except Exception as e:
        print(f"⚠️  Failed to save comprehensive analysis CSV: {e}")
    try:
        md = "# Top-10 MOF comprehensive analysis table\n\n" + _dataframe_to_markdown(analysis_df)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"📝 Comprehensive analysis Markdown saved: {md_path}")
    except Exception as e:
        print(f"⚠️  Failed to save comprehensive analysis Markdown: {e}")
    return md_path, csv_path

# ===== End of comprehensive analysis table helpers =====


# ===== ABDF dataset preparation helpers =====
def _json_records_from_file(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read JSON: {e}")
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if 'results' in data and isinstance(data['results'], list):
            return data['results']
        if all(isinstance(v, dict) for v in data.values()):
            rows = []
            for k, v in data.items():
                row = v.copy()
                if not any(key in row for key in ['id', 'ID', 'name', 'material_name']):
                    row['id'] = k
                else:
                    row['_key'] = k
                rows.append(row)
            return rows
        return [data]
    return []

def _find_join_keys(df1: pd.DataFrame, df2: pd.DataFrame):
    cand1 = [c for c in ['material_name', 'ID', 'id', 'name'] if c in df1.columns]
    cand2 = [c for c in ['material_name', 'ID', 'id', 'name'] if c in df2.columns]
    best = None
    best_score = -1
    cache2 = {c: set(df2[c].astype(str).dropna().unique()) for c in cand2}
    for c1 in cand1:
        s1 = set(df1[c1].astype(str).dropna().unique())
        for c2 in cand2:
            inter = len(s1 & cache2[c2])
            if inter > best_score:
                best = (c1, c2)
                best_score = inter
    return best

def _apply_metric_synonyms(df: pd.DataFrame) -> pd.DataFrame:
    syns = {
        'purity': ['purity', 'Purity', 'pur', 'purity_pct', 'Purity_%'],
        'recovery': ['recovery', 'Recovery', 'rec', 'yield', 'Yield'],
        'efficiency': ['efficiency', 'Efficiency', 'eff', 'score', 'fitness', 'overall_score']
    }
    for target, cands in syns.items():
        if target in df.columns:
            continue
        for c in cands:
            if c in df.columns:
                df = df.rename(columns={c: target})
                break
    return df

def _ensure_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    if 'efficiency' not in df.columns:
        if 'purity' in df.columns and 'recovery' in df.columns:
            try:
                p = pd.to_numeric(df['purity'], errors='coerce')
                r = pd.to_numeric(df['recovery'], errors='coerce')
                p = p / 100.0 if p.max(skipna=True) and p.max(skipna=True) > 1.5 else p
                r = r / 100.0 if r.max(skipna=True) and r.max(skipna=True) > 1.5 else r
                df['efficiency'] = p * r
            except Exception:
                pass
    return df

def load_abdf_dataset(features_csv: str, results_json: str) -> pd.DataFrame:
    print("📥 Loading ABDF dataset...")
    try:
        fdf = pd.read_csv(features_csv)
        print(f"📄 Feature CSV: {features_csv} shape: {fdf.shape}")
    except Exception as e:
        print(f"❌ Failed to load feature CSV: {e}")
        return None
    recs = _json_records_from_file(results_json)
    if not recs:
        print("❌ Failed to load results JSON or file is empty")
        return None
    rdf = pd.json_normalize(recs)
    print(f"📄 Results JSON: {results_json} shape: {rdf.shape}")
    rdf = _apply_metric_synonyms(rdf)
    join = _find_join_keys(fdf, rdf)
    if not join:
        print("❌ Could not determine join keys (material_name/ID/id/name)")
        return None
    left_key, right_key = join
    merged = pd.merge(fdf, rdf, left_on=left_key, right_on=right_key, how='inner')
    print(f"🔗 Merged dataset shape: {merged.shape} (on {left_key} ~ {right_key})")
    merged = _apply_metric_synonyms(merged)
    merged = _ensure_efficiency(merged)
    required_cols = ["purity", "recovery", "efficiency"]
    missing = [c for c in required_cols if c not in merged.columns]
    if missing:
        print(f"❌ ABDF merge missing required columns: {missing}")
        return None
    return merged

def prepare_candidate_data_from_df(df: pd.DataFrame, k_screen=50):
    print("📊 Preparing candidates using the ABDF dataset")
    try:
        required_cols = ["purity", "recovery", "efficiency"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None
        print("🔧 Performing data pre-optimisation...")
        keep_cols = []
        for id_col in ['material_name', 'ID', 'id', 'name']:
            if id_col in df.columns:
                keep_cols.append(id_col)
                break
        keep_cols.extend([col for col in required_cols if col in df.columns])
        remaining_cols = [col for col in df.columns if col not in keep_cols]
        valuable_cols = []
        for col in remaining_cols[:20]:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if df[col].nunique() > 1 and not df[col].isna().all():
                    valuable_cols.append(col)
                    if len(valuable_cols) >= 10:
                        break
        keep_cols.extend(valuable_cols)
        # Always keep key CIF structural descriptors when available
        cif_core_cols = [
            'max_pore_diameter', 'largest_free_sphere', 'density', 'cell_volume',
            'element_diversity', 'metal_fraction', 'carbon_content', 'cell_sphericity',
            'cell_length_ratio_max', 'avg_electronegativity', 'electronegativity_std'
        ]
        for c in cif_core_cols:
            if c in df.columns and c not in keep_cols:
                keep_cols.append(c)
        df = df[keep_cols].copy()
        print(f"📊 Data pre-optimisation retained {len(keep_cols)} columns")
        print("🔧 Running baseline filtering...")
        before_count = len(df)
        df = df.dropna(subset=required_cols)
        after_count = len(df)
        if before_count > after_count:
            print(f"⚠️  Removed {before_count - after_count} rows with missing required values")
        df = df[df['efficiency'] >= 0.01]
        final_count = len(df)
        print(f"📊 Data after filtering: {final_count} rows")
        df = df.sort_values("efficiency", ascending=False).head(k_screen).reset_index(drop=True)
        df['screen_rank'] = range(1, len(df) + 1)
        print(f"🎯 Final candidate dataset: {len(df)} rows × {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"❌ ABDF candidate preparation failed: {str(e)}")
        return None

# ===== End of ABDF dataset helpers =====


def _dump_cfg(tag, cfgobj):
    print(f"\n[{tag}] FINAL LLM CONFIG >>>")
    print(json.dumps(cfgobj, ensure_ascii=False, indent=2))


def create_agents(k_screen=50, n_final=10):
    print("🤖 Initialising agents...")

    # Shared configuration (ensure thinking disabled and non-streaming)
    gpt5_cfg = cfg("gpt-5", 0.25)
    _dump_cfg("GPT5", gpt5_cfg)

    gemini_cfg = cfg("gemini-2.5-pro", 0.25)
    _dump_cfg("GEMINI", gemini_cfg)

    o3mini_cfg = cfg("o3-mini", 0.2)
    _dump_cfg("O3_MINI", o3mini_cfg)

    screening = autogen.AssistantAgent(
        name="Screening",
        llm_config=copy.deepcopy(gemini_cfg),
        system_message=load_prompt("screening.txt")
    )
    chemist = autogen.AssistantAgent(
        name="Chemist",
        llm_config=copy.deepcopy(gemini_cfg),
        system_message=load_prompt("chemist.txt")
    )
    engineer = autogen.AssistantAgent(
        name="Engineer",
        llm_config=copy.deepcopy(gemini_cfg),
        system_message=load_prompt("engineer.txt")
    )
    officer = autogen.AssistantAgent(
        name="Compliance",
        llm_config=copy.deepcopy(gemini_cfg),
        system_message=load_prompt("compliance.txt")
    )
    moderator = autogen.AssistantAgent(
        name="Moderator",
        llm_config=copy.deepcopy(gpt5_cfg),  # switch to gemini_cfg if compatibility issues arise
        system_message=load_prompt("moderator.txt")
    )
    debater = autogen.AssistantAgent(
        name="Debater",
        llm_config=copy.deepcopy(o3mini_cfg),  # switch to gemini_cfg if compatibility issues arise
        system_message="You are GPT-o3, master of deep debate."
    )

    return screening, chemist, engineer, officer, moderator, debater

def run_multi_agent_screening(candidate_df, k_screen=50, n_final=10):
    """Run the multi-agent screening workflow."""
    print("🚀 Launching multi-agent screening...")
    
    # Initialise agent cohort
    screening, chemist, engineer, officer, moderator, debater = create_agents(k_screen, n_final)
    
    # Limit how many rows are sent to the Screening agent
    max_screening_rows = min(k_screen, 10)
    print(f"📊 Data size control: sending first {max_screening_rows} rows")

    # Prepare identifier column and candidate IDs for scoring/visualisation
    id_col = get_identifier_col(candidate_df)
    candidate_ids = [str(x) for x in candidate_df[id_col].head(max_screening_rows).tolist()]
    scoring_instruction = build_scoring_instruction(id_col, candidate_ids)
    
    # Render compact table for the prompt
    table_markdown = df_to_markdown_table(candidate_df, max_cols=16, max_rows=max_screening_rows)
    
    print("📤 Sending candidate summary to the screening analyst...")
    
    # Compose the initial system prompt dispatched to the screening analyst
    initial_message = f"""Please analyse the following MOF candidates:

{table_markdown}

Work through the collaboration pipeline in the order below:
1. Screening Analyst – initial triage
2. Chemist – chemical suitability
3. Engineer – process and economic assessment
4. Compliance – regulatory and safety review
5. Moderator – consolidates the final recommendation
6. (Optional) Debater – deep-dive rebuttal if required

Objective: shortlist the best {n_final} MOF candidates from this pool of {max_screening_rows}.

Adhere strictly to the scoring specification (used downstream for visualisation):
{scoring_instruction}
"""
    
    try:
        print("💬 Starting multi-agent dialogue...")
        print("=" * 80)
        
        # Bootstrap the user proxy agent (no interactive input)
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )
        
        # Sequentially interact with each agent
        messages = []
        
        # Step 1: Screening analyst
        print("🔍 [Screening Analyst] starting analysis...")
        result = user_proxy.initiate_chat(
            screening, 
            message=initial_message,
            max_turns=1
        )
        screening_response = result.chat_history[-1]['content'] if result.chat_history else "No response"
        messages.append({"name": "Screening", "content": screening_response})
        
        # Step 2: Chemist
        print("⚗️ [Chemist] starting assessment...")
        chemist_msg = (
            "Based on the screening analyst's findings:\n"
            f"{screening_response}\n\n"
            "Please evaluate these MOF materials from a chemistry perspective.\n\n"
            f"Please follow the same scoring output format:\n{scoring_instruction}"
        )
        result = user_proxy.initiate_chat(
            chemist, 
            message=chemist_msg,
            max_turns=1
        )
        chemist_response = result.chat_history[-1]['content'] if result.chat_history else "No response"
        messages.append({"name": "Chemist", "content": chemist_response})
        
        # Step 3: Process engineer
        print("🔧 [Engineer] starting assessment...")
        engineer_msg = (
            "Building on the prior assessments:\n\n"
            f"Screening analyst: {screening_response}\n\n"
            f"Chemist: {chemist_response}\n\n"
            "Please evaluate these MOF materials from a process economics perspective.\n\n"
            f"Please follow the same scoring output format:\n{scoring_instruction}"
        )
        result = user_proxy.initiate_chat(
            engineer, 
            message=engineer_msg,
            max_turns=1
        )
        engineer_response = result.chat_history[-1]['content'] if result.chat_history else "No response"
        messages.append({"name": "Engineer", "content": engineer_response})
        
        # Step 4: Compliance officer
        print("📋 [Compliance Officer] starting assessment...")
        compliance_msg = (
            "With the previous analyses in mind, please assess compliance risk:\n\n"
            f"Engineer evaluation: {engineer_response}\n\n"
            "Provide a compliance assessment of the MOF candidates.\n\n"
            f"Please follow the same scoring output format:\n{scoring_instruction}"
        )
        result = user_proxy.initiate_chat(
            officer, 
            message=compliance_msg,
            max_turns=1
        )
        compliance_response = result.chat_history[-1]['content'] if result.chat_history else "No response"
        messages.append({"name": "Compliance", "content": compliance_response})
        
        # Step 5: Moderator synthesises the recommendation
        print("⚖️ [Moderator] compiling final decision...")
        moderator_msg = (
            "Consolidate the experts' viewpoints and produce the final decision:\n\n"
            f"Screening analyst: {screening_response}\n\n"
            f"Chemist: {chemist_response}\n\n"
            f"Engineer: {engineer_response}\n\n"
            f"Compliance officer: {compliance_response}\n\n"
            f"Select the best {n_final} candidates from these {max_screening_rows} MOFs "
            "and present the final recommendations in a table."
        )
        
        result = user_proxy.initiate_chat(
            moderator, 
            message=moderator_msg,
            max_turns=1
        )
        moderator_response = result.chat_history[-1]['content'] if result.chat_history else "No response"
        messages.append({"name": "Moderator", "content": moderator_response})
        
        # Create a lightweight chat object compatible with downstream handling
        class MockGroupChat:
            def __init__(self):
                self.messages = [
                    {"name": "System", "content": initial_message}
                ] + messages
                # Persist derived artefacts for later reporting
                self.candidate_ids = candidate_ids
                try:
                    self.scores_df = build_scores_matrix(messages, candidate_ids)
                except Exception as _e:
                    print(f"⚠️ Failed to build score matrix: {_e}")
                    self.scores_df = pd.DataFrame()
                # Snapshot for final reporting
                self.id_col = id_col
                try:
                    self.candidate_df_top = candidate_df.head(max_screening_rows).copy()
                except Exception:
                    self.candidate_df_top = None
        
        groupchat = MockGroupChat()
        
        # Display conversation summary
        print("\n" + "=" * 80)
        print("📜 Conversation transcript:")
        print("=" * 80)
        
        for i, message in enumerate(groupchat.messages):
            speaker = message.get('name', 'Unknown')
            content = message.get('content', '')
            
            # Speaker labels
            if speaker == 'System':
                print("\n🤖 [System]")
            elif speaker == 'Screening':
                print("\n🔍 [Screening Analyst]")
            elif speaker == 'Chemist':
                print("\n⚗️ [Chemist]")
            elif speaker == 'Engineer':
                print("\n🔧 [Engineer]")
            elif speaker == 'Compliance':
                print("\n📋 [Compliance Officer]")
            elif speaker == 'Moderator':
                print("\n⚖️ [Moderator]")
            elif speaker == 'Debater':
                print("\n💭 [Debater]")
            else:
                print(f"\n👤 [{speaker}]")
            
            # Clip long outputs for terminal readability
            if len(content) > 800:
                print(f"{content[:800]}...")
                print("[Content truncated; see saved file for the full transcript]")
            else:
                print(content)
            
            print("-" * 60)
        
        # Show derived score matrix if available
        try:
            if hasattr(groupchat, 'scores_df') and not groupchat.scores_df.empty:
                print("\n📊 Score matrix (rows = agents, columns = candidates):")
                print(groupchat.scores_df)
        except Exception:
            pass
        
        print(f"\n✅ Dialogue finished with {len(groupchat.messages)} messages")
        
        return groupchat
        
    except Exception as e:
        print(f"❌ Multi-agent screening failed: {str(e)}")
        return None

def save_results(groupchat, output_dir="results"):
    """Persist transcripts, score matrices, and summary artefacts."""
    print("💾 Saving screening outputs...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Persist the full conversation log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    chat_file = output_path / f"llm_agent_screening_chat_{timestamp}.txt"
    with open(chat_file, 'w', encoding='utf-8') as f:
        f.write("Multi-agent MOF screening transcript\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, message in enumerate(groupchat.messages):
            f.write(f"[{i+1}] {message.get('name', 'Unknown')}: \n")
            f.write(message.get('content', '') + "\n")
            f.write("-" * 40 + "\n")
    
    print(f"📋 Conversation log saved to: {chat_file}")

    # Persist score matrices and heatmaps
    try:
        if hasattr(groupchat, 'scores_df') and not groupchat.scores_df.empty:
            fig_path, csv_path = save_scores_heatmap(groupchat.scores_df, output_dir, timestamp)
            print(f"🗺️ Score heatmap path: {fig_path}")
        else:
            print("⚠️  Score matrix unavailable (agents may not have emitted JSON scores)")
    except Exception as e:
        print(f"⚠️  Failed to save score heatmap: {e}")

    # Produce top-10 comprehensive analysis tables
    try:
        if hasattr(groupchat, 'scores_df') and hasattr(groupchat, 'candidate_df_top') and hasattr(groupchat, 'id_col'):
            if groupchat.candidate_df_top is not None and not groupchat.scores_df.empty:
                analysis_df = build_comprehensive_analysis_table(groupchat.candidate_df_top, groupchat.scores_df, groupchat.id_col, top_k=10)
                if analysis_df is not None and not analysis_df.empty:
                    md_path, comp_csv_path = save_comprehensive_analysis(analysis_df, output_dir, timestamp)
                else:
                    print("⚠️  Comprehensive analysis table is empty (missing scores or candidate data)")
        else:
            print("⚠️  Missing data required to generate the analysis table")
    except Exception as e:
        print(f"⚠️  Failed to save comprehensive analysis table: {e}")
    
    # Attempt to extract the moderator's final decision  
    moderator_result = None
    for message in reversed(groupchat.messages):
        if message.get('name') == 'Moderator':
            content = message.get('content', '')
            if '|' in content and 'Rank' in content:  # crude table detection
                moderator_result = content
                break
    
    if moderator_result:
        result_file = output_path / f"llm_agent_final_selection_{timestamp}.md"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("# Multi-agent screening final outcome\n\n")
            f.write(f"**Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Final recommended materials\n\n")
            f.write(moderator_result)
        
        print(f"🎯 Final screening output saved to: {result_file}")
        return result_file
    else:
        print("⚠️  Moderator did not supply a final decision")
        return chat_file

def main():
    """Command-line entry point"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Multi-agent MOF material screening system')
    parser.add_argument('--input-file', '-i', type=str, 
                       help='Path to a candidate file for screening')
    parser.add_argument('--k-screen', type=int, default=20,
                       help='Number of candidates to pre-screen (default: 20)')
    parser.add_argument('--n-final', type=int, default=3,
                       help='Number of final recommendations (default: 3)')
    # Optional ABDF dataset arguments
    parser.add_argument('--abdf-features', type=str, default=str(PROJECT_ROOT / 'data' / 'cif_features' / 'cif_features_preprocessed.csv'),
                       help='Path to ABDF features CSV')
    parser.add_argument('--abdf-results', type=str, default=str(PROJECT_ROOT / 'data' / 'TSA_results' / 'uff_ddec_all_results_20250801_182548.json'),
                       help='Path to ABDF results JSON')
    parser.add_argument('--use-abdf', action='store_true', help='Prefer using the ABDF dataset for screening')
    
    args = parser.parse_args()
    
    print("🎯 Multi-agent MOF material screening system ready")
    print(f"📂 Working directory: {os.getcwd()}")
    
    if args.input_file:
        print(f"📋 Specified input file: {args.input_file}")
    
    # Validate runtime environment
    if not check_environment():
        print("❌ Environment check failed; verify configuration")
        return False
    
    # Attempt to use the ABDF dataset first
    candidate_df = None
    try:
        feats_p = Path(args.abdf_features) if args.abdf_features else None
        results_p = Path(args.abdf_results) if args.abdf_results else None
        auto_detect = feats_p and results_p and feats_p.exists() and results_p.exists()
        if args.use_abdf or auto_detect:
            print("🧩 Screening with the ABDF dataset")
            abdf_df = load_abdf_dataset(str(feats_p), str(results_p))
            if abdf_df is not None and not abdf_df.empty:
                candidate_df = prepare_candidate_data_from_df(abdf_df, args.k_screen)
                if candidate_df is None:
                    print("⚠️ ABDF preparation failed; falling back to candidate-file mode")
            else:
                print("⚠️ ABDF dataset load failed; falling back to candidate-file mode")
    except Exception as e:
        print(f"⚠️ ABDF dataset processing error: {e}")

    # Fall back to candidate files if ABDF data is unavailable
    if candidate_df is None:
        # Locate candidate file
        candidate_file = find_candidate_files(args.input_file)
        if candidate_file is None:
            return False
        # Capture configuration values
        K_SCREEN = args.k_screen  # number of candidates to pre-screen
        N_FINAL = args.n_final   # number of final recommendations
        print(f"🎯 Pre-screening count: {K_SCREEN}")
        print(f"🎯 Final recommendation count: {N_FINAL}")
        # Prepare candidate data
        candidate_df = prepare_candidate_data(candidate_file, K_SCREEN)
        if candidate_df is None:
            return False
    else:
        # ABDF dataset already loaded
        K_SCREEN = args.k_screen
        N_FINAL = args.n_final
        print(f"🎯 Pre-screening count: {K_SCREEN}")
        print(f"🎯 Final recommendation count: {N_FINAL}")
    
    # Run the multi-agent workflow
    groupchat = run_multi_agent_screening(candidate_df, K_SCREEN, N_FINAL)
    if groupchat is None:
        return False
    
    # Persist outputs
    result_file = save_results(groupchat)
    
    # Present summary
    print("\n" + "=" * 80)
    print("🎉 Multi-agent screening complete")
    print("=" * 80)
    print(f"📊 Input candidate count: {len(candidate_df)}")
    print(f"🎯 Target screening count: {N_FINAL}")
    print(f"💬 Dialogue turns: {len(groupchat.messages) if groupchat else 0}")
    print(f"📁 Output file: {result_file}")
    
    print("\n💡 Next steps:")
    print("1. Review the final screening output")
    print("2. Inspect each agent's rationale")
    print("3. Adjust screening parameters and rerun if necessary")
    print("4. Select recommended materials for experimental validation")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 

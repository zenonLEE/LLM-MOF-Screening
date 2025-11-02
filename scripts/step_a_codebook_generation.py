#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step A: Build the code-book by merging CIF-derived features with TSA metrics
into a unified analysis-ready table.
"""

import json
import re
from pathlib import Path
import pandas as pd

# ---------- config ----------
BASE_DIR = Path(__file__).resolve().parents[1]
FEAT_PATH = BASE_DIR / "data" / "cif_features" / "cif_features_preprocessed.csv"
TSA_PATH = BASE_DIR / "data" / "TSA_results" / "uff_ddec_all_results_20250801_182548.json"
OUT_PARQUET = BASE_DIR / "data" / "codebook.parquet"
# ----------------------------

def strip_prefix(name: str) -> str:
    """Remove known prefixes from a material identifier."""
    return re.sub(r"^(DDEC_|EQeq_|Qeq_|NEUTRAL_|MPNN_|PACMOF_)", "", name)

def main():
    print("🚀 Building code-book...")
    
    # Step A1 — load feature table
    print(f"📖 Loading feature file: {FEAT_PATH}")
    if not FEAT_PATH.exists():
        raise FileNotFoundError(f"Feature file not found: {FEAT_PATH}")
    
    feat = pd.read_csv(FEAT_PATH)
    print(f"✅ Feature table loaded, shape: {feat.shape}")
    
    # Clean material names
    feat["clean_material_name"] = feat["material_name"].apply(strip_prefix)
    print("📝 Material name prefixes removed")
    
    # Step A2 — load TSA results
    print(f"📖 Loading TSA results: {TSA_PATH}")
    if not TSA_PATH.exists():
        raise FileNotFoundError(f"TSA result file not found: {TSA_PATH}")

    with open(TSA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    # Adapt to JSON structure variants
    if isinstance(raw, dict) and "results" in raw:
        tsa_data = raw["results"]
    else:
        tsa_data = raw
    
    tsa = pd.json_normalize(tsa_data)
    print(f"✅ TSA results parsed, shape: {tsa.shape}")
    
    # Align column names
    if "material" in tsa.columns:
        tsa.rename(columns={"material": "material_name"}, inplace=True)
    
    # Clean material names
    tsa["clean_material_name"] = tsa["material_name"].apply(strip_prefix)
    
    # Ensure critical columns exist
    required_cols = ["purity", "recovery", "efficiency"]
    missing_cols = [col for col in required_cols if col not in tsa.columns]
    if missing_cols:
        print(f"⚠️  Warning: missing columns in TSA data: {missing_cols}")
        print(f"📋 Available columns: {list(tsa.columns)}")
    
    # Step A3 — merge features and TSA metrics
    print("🔗 Merging feature and TSA datasets...")
    codebook = pd.merge(
        feat, 
        tsa[["clean_material_name"] + [col for col in required_cols if col in tsa.columns]],
        on="clean_material_name", 
        how="inner"
    )
    
    print("✅ Merge complete")
    print(f"📊 Feature rows: {feat.shape[0]}")
    print(f"📊 TSA rows: {tsa.shape[0]}")
    print(f"📊 Merged rows: {codebook.shape[0]}")
    
    # Persist output
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    codebook.to_parquet(OUT_PARQUET, index=False)
    print(f"💾 Code-book saved to: {OUT_PARQUET}")
    print(f"📋 Final shape: {codebook.shape}")
    
    # Summary stats
    print("\n📈 Summary statistics:")
    print(f"Column count: {codebook.shape[1]}")
    if 'purity' in codebook.columns:
        print(f"Purity range: {codebook['purity'].min():.4f} - {codebook['purity'].max():.4f}")
    if 'recovery' in codebook.columns:
        print(f"Recovery range: {codebook['recovery'].min():.4f} - {codebook['recovery'].max():.4f}")
    if 'efficiency' in codebook.columns:
        print(f"Efficiency range: {codebook['efficiency'].min():.4f} - {codebook['efficiency'].max():.4f}")
    
    print("\n✅ Code-book ready!")
    return codebook

if __name__ == "__main__":
    codebook = main() 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step F: LLM multi-agent screening.

Orchestrates a panel of LLM agents to review and prioritise candidate MOF
materials.
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path

def main():
    """Entry point for invoking the multi-agent screening phase."""
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Step F: LLM multi-agent screening")
    parser.add_argument('--input-file', '-i', type=str,
                       help='Optional path to a candidate CSV for screening')
    parser.add_argument('--k-screen', type=int, default=20,
                       help='Number of candidates to pre-screen (default: 20)')
    parser.add_argument('--n-final', type=int, default=3,
                       help='Number of final recommendations (default: 3)')
    
    args = parser.parse_args()
    
    print("🤖 Step F: LLM multi-agent screening started")
    print("="*60)
    
    # Ensure run_agents.py is available
    script_path = Path(__file__).resolve().parent / "run_agents.py"
    if not script_path.exists():
        print("❌ run_agents.py not found")
        print("💡 Ensure you run this from the MOFSelect project root")
        return False
    
    try:
        # Build CLI arguments
        cmd = [sys.executable, str(script_path)]
        if args.input_file:
            cmd.extend(["--input-file", args.input_file])
        cmd.extend(["--k-screen", str(args.k_screen)])
        cmd.extend(["--n-final", str(args.n_final)])
        
        # Invoke run_agents.py
        print("🚀 Launching multi-agent screening workflow...")
        if args.input_file:
            print(f"📋 Using specified input file: {args.input_file}")
        print(f"🎯 Pre-screen count: {args.k_screen}, final recommendations: {args.n_final}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.stdout:
            print("📤 Agent screening output:")
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("⚠️  Error output:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ LLM multi-agent screening succeeded")
            return True
        else:
            print("\n❌ LLM multi-agent screening failed")
            return False
            
    except Exception as e:
        print(f"❌ Execution error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 

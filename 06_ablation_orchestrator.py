"""
06_ablation_orchestrator.py
===========================
Orchestrator for 6-phase factorial ablation study with support for
diagnostic Mode C (Gold Context Upper Bound) and isolated memory management.

Experimental configurations:
0. Mode C: Gold Context Upper Bound (diagnostic LLM ceiling on gold context).
1. Flat RAG (Baseline): Standard retrieval over raw leaf snippets.
2. Standard RAPTOR: Classical hierarchical tree without entity preservation.
3. RAPTOR + Entity Preservation (EP): Hierarchical tree with E-A-R triplet injection.
4. RAPTOR + Adaptive Router: Tree with dynamic layer selection (LightGBM).
5. RAPTOR + Oracle Router: Theoretical upper bound on routing (l*).
6. Combined System: Final proposed method (RAPTOR + EP + Adaptive Router).
"""

import os
import sys
import subprocess
import argparse
import time

CONFIGS = [
    {
        "id": 0,
        "name": "Mode C: Gold Context (LLM Ceiling)",
        "tree_type": "standard",
        "mode_c": True,
        "desc": "Diagnostic mode: passing reference (Gold) snippets directly into LLM."
    },
    {
        "id": 1,
        "name": "Flat RAG (Baseline)",
        "tree_type": "standard",
        "mode_c": False,
        "desc": "Dense retrieval MedCPT + FAISS over leaf passages (Leaf level)."
    },
    {
        "id": 2,
        "name": "Standard RAPTOR",
        "tree_type": "standard",
        "mode_c": False,
        "desc": "RAPTOR without entity preservation, cross-layer retrieval."
    },
    {
        "id": 3,
        "name": "RAPTOR + Entity Preservation (EP)",
        "tree_type": "ep",
        "mode_c": False,
        "desc": "RAPTOR with mandatory E-A-R triplet preservation during summarization."
    },
    {
        "id": 4,
        "name": "RAPTOR + Adaptive Router",
        "tree_type": "standard",
        "mode_c": False,
        "desc": "RAPTOR with granularity level selection via LightGBM classifier."
    },
    {
        "id": 5,
        "name": "RAPTOR + Oracle Router (Upper Bound)",
        "tree_type": "standard",
        "mode_c": False,
        "desc": "RAPTOR with theoretical best level selection via Utility Function."
    },
    {
        "id": 6,
        "name": "Combined System (Final Proposed Method)",
        "tree_type": "ep",
        "mode_c": False,
        "desc": "RAPTOR + Entity Preservation + Adaptive Granularity Routing."
    }
]

def run_isolated_command(cmd: list) -> bool:
    """Run execution step in an isolated subprocess with streaming output."""
    cmd_str = " ".join(cmd)
    print(f"\n[ORCHESTRATOR] >>> Launching: {cmd_str}", flush=True)
    start_time = time.time()
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n[ORCHESTRATOR] ERROR in {cmd[1]} (exit code {result.returncode})\n", flush=True)
        return False
        
    print(f"[ORCHESTRATOR] <<< Completed successfully in {elapsed:.2f} s.\n", flush=True)
    return True

def run_pipeline(seed: int = 42, limit: int = 100, force_rebuild: bool = False):
    print("=" * 70)
    print("  STARTING BIOASQ ABLATION STUDY BENCHMARK")
    print("=" * 70)
    
    py = sys.executable
    results_csv = "results_ablation.csv"
    
    if os.path.exists(results_csv):
        os.remove(results_csv)

    # ----------------------------------------------------
    # PHASE 1: Data Preparation and Splits
    # ----------------------------------------------------
    print("\n--- PHASE 1: Data Loading & Splitting ---")
    data_cmd = [py, "01_data_loader.py", "--seed", str(seed)]
    if limit > 0:
        data_cmd.extend(["--limit", str(limit)])
    if not run_isolated_command(data_cmd):
        return

    # ----------------------------------------------------
    # PHASE 2: Deterministic E-A-R Triplet Extraction
    # ----------------------------------------------------
    print("\n--- PHASE 2: Biomedical Entity & Relation Extraction (E-A-R) ---")
    if not run_isolated_command([py, "02_ear_extractor.py"]):
        return

    # ----------------------------------------------------
    # PHASE 3: RAPTOR Tree Construction (Standard & EP with Caching)
    # ----------------------------------------------------
    print("\n--- PHASE 3: RAPTOR Tree Construction ---")
    tree_std_cmd = [
        py, "03_raptor_builder.py",
        "--input", "data/splits/test_ear.json",
        "--output", "data/trees/test_tree_standard.json"
    ]
    if force_rebuild: tree_std_cmd.append("--force_rebuild")
    if not run_isolated_command(tree_std_cmd):
        return

    tree_ep_cmd = [
        py, "03_raptor_builder.py",
        "--input", "data/splits/test_ear.json",
        "--output", "data/trees/test_tree_ep.json",
        "--preserve_ear"
    ]
    if force_rebuild: tree_ep_cmd.append("--force_rebuild")
    if not run_isolated_command(tree_ep_cmd):
        return

    dev_tree_cmd = [
        py, "03_raptor_builder.py",
        "--input", "data/splits/dev_ear.json",
        "--output", "data/trees/dev_tree.json"
    ]
    if force_rebuild: dev_tree_cmd.append("--force_rebuild")
    run_isolated_command(dev_tree_cmd)

    # ----------------------------------------------------
    # PHASE 4: Granularity Router Training
    # ----------------------------------------------------
    print("\n--- PHASE 4: Adaptive Router Training & Oracle Calculation on Dev-set ---")
    run_isolated_command([
        py, "04_granularity_router.py",
        "--dev_input", "data/trees/dev_tree.json",
        "--model_out", "models/router_lgbm.pkl"
    ])

    # ----------------------------------------------------
    # PHASE 5: Running Experimental Configurations (C0 + C1-C6)
    # ----------------------------------------------------
    print("\n--- PHASE 5: Running Experimental Configurations ---")
    total_cfgs = len(CONFIGS)
    for idx, cfg in enumerate(CONFIGS, 1):
        cfg_id = cfg["id"]
        cfg_name = cfg["name"]
        print("\n" + "=" * 65)
        print(f"  [{idx}/{total_cfgs}] CONFIGURATION {cfg_id}: {cfg_name}")
        print(f"  Description: {cfg['desc']}")
        print("=" * 65, flush=True)

        tree_file = "data/trees/test_tree_ep.json" if cfg["tree_type"] == "ep" else "data/trees/test_tree_standard.json"
        res_file = f"data/results_config_{cfg_id}.json"

        # 5.1 Retrieval and Generation
        ret_cmd = [
            py, "05_retrieval_and_generation.py",
            "--config_id", str(cfg_id),
            "--input", tree_file,
            "--output", res_file
        ]
        if cfg.get("mode_c"):
            ret_cmd.append("--mode_c")

        ret_ok = run_isolated_command(ret_cmd)
        if not ret_ok:
            print(f"[!] Failure at retrieval and generation phase for Config {cfg_id}.")
            continue

        # 5.2 Metric Evaluation (BioASQ, E-A-R Loss, RAGAS)
        eval_ok = run_isolated_command([
            py, "07_evaluation_metrics.py",
            "--config_id", str(cfg_id),
            "--input_results", res_file,
            "--csv_out", results_csv
        ])
        if not eval_ok:
            print(f"[!] Failure at metric evaluation phase for Config {cfg_id}.")
            continue

    # ----------------------------------------------------
    # PHASE 6: Publication Plots & Reports Generation
    # ----------------------------------------------------
    print("\n--- PHASE 6: Generating Publication Plots, LaTeX Table & Significance Tests ---")
    run_isolated_command([
        py, "08_visualization.py",
        "--csv_file", results_csv,
        "--out_dir", "plots"
    ])

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"  1. Detailed metrics data: {results_csv}")
    print("  2. High-res plots (300 DPI): plots/")
    print("  3. Summary report: plots/ablation_summary_table.md")
    print("  4. LaTeX table: plots/ablation_table.tex")
    print("  5. Statistical significance report: plots/statistical_significance_report.md")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Orchestrator.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit", type=int, default=100, help="Number of questions for experiment (0 = all)")
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild RAPTOR trees")
    args = parser.parse_args()
    
    run_pipeline(seed=args.seed, limit=args.limit, force_rebuild=args.force_rebuild)


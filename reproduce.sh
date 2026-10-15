#!/usr/bin/env bash
# ==============================================================================
# Master Reproduction Script for Biomedical Adaptive Granularity RAG Benchmark
# ==============================================================================
set -e

# Detect python interpreter (respect active env, fallback to python3 or .venv)
if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "[-] Error: Python interpreter not found. Please install Python 3.10+."
    exit 1
fi

print_header() {
    echo ""
    echo "================================================================================"
    echo " $1"
    echo "================================================================================"
}

show_help() {
    echo "Usage: ./reproduce.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  --eval      (Default) Fast statistical evaluation & visualization from precomputed"
    echo "              ablation results (07_evaluation_metrics.py + 08_visualization.py)."
    echo "              Takes < 5 seconds. Requires no GPU and no API keys."
    echo ""
    echo "  --full      End-to-end reproduction from scratch (Stages 01 through 08)."
    echo "              Requires GROQ_API_KEY in .env, PyTorch, and MedCPT encoders."
    echo ""
    echo "  --help      Show this help message."
    echo ""
}

run_eval() {
    print_header "Stage 1/2: Quantitative Metric Evaluation (07_evaluation_metrics.py)"
    rm -f results_ablation.csv
    for cfg in {0..6}; do
        if [ -f "data/results_config_${cfg}.json" ]; then
            $PYTHON 07_evaluation_metrics.py \
                --config_id "$cfg" \
                --input_results "data/results_config_${cfg}.json" \
                --csv_out "results_ablation.csv"
        else
            echo "[-] Warning: data/results_config_${cfg}.json not found."
        fi
    done

    print_header "Stage 2/2: Publication Figures & Academic Reports (08_visualization.py)"
    $PYTHON 08_visualization.py --csv_file results_ablation.csv --out_dir plots

    print_header "Evaluation & Visualization Complete!"
    echo "[+] All publication figures (PNG, PDF, SVG) and tables are saved in 'plots/'."
    echo "[+] Summary table:               plots/ablation_summary_table.md"
    echo "[+] Statistical test report:     plots/statistical_significance_report.md"
    echo "[+] LaTeX table:                 plots/ablation_table.tex"
}

run_full() {
    if [ ! -f ".env" ]; then
        if [ -z "$GROQ_API_KEY" ]; then
            echo "[-] Error: Neither .env file nor GROQ_API_KEY environment variable found."
            echo "    Please copy .env.example to .env and configure your GROQ_API_KEY:"
            echo "    cp .env.example .env"
            exit 1
        fi
    fi

    print_header "Stage 1/8: Data Partitioning (01_data_loader.py)"
    $PYTHON 01_data_loader.py

    print_header "Stage 2/8: E-A-R Extraction (02_ear_extractor.py)"
    $PYTHON 02_ear_extractor.py

    print_header "Stage 3/8: RAPTOR Tree Construction (03_raptor_builder.py)"
    $PYTHON 03_raptor_builder.py --build_both

    print_header "Stage 4/8: Granularity Router Training (04_granularity_router.py)"
    $PYTHON 04_granularity_router.py

    print_header "Stage 5/8: End-to-End Ablation (06_ablation_orchestrator.py)"
    $PYTHON 06_ablation_orchestrator.py

    print_header "Stage 6/8: Evaluation & Significance Testing (07_evaluation_metrics.py)"
    $PYTHON 07_evaluation_metrics.py

    print_header "Stage 7/8: Visualization & Reports (08_visualization.py)"
    $PYTHON 08_visualization.py --csv_file results_ablation.csv --out_dir plots

    print_header "Full Pipeline Execution Successfully Completed!"
}

# Argument dispatch
MODE="${1:---eval}"

case "$MODE" in
    --eval)
        run_eval
        ;;
    --full)
        run_full
        ;;
    --help|-h)
        show_help
        ;;
    *)
        echo "[-] Unknown option: $MODE"
        show_help
        exit 1
        ;;
esac

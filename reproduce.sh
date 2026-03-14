#!/bin/bash
# Quick validation workflow for reviewers (10 minutes)
# Assumes results/ directory contains pre-computed results.json files
set -e

echo "🔬 BitNet Reproducibility Validation..."
echo ""
echo "This script validates that paper artifacts can be regenerated from results/"
echo "Full experimental reproduction requires 920 GPU-hours (see EXPERIMENTS_REFERENCE.sh)"
echo ""

# Setup environment
echo "📦 Setting up environment..."
uv sync
echo "✓ Environment ready"
echo ""

# Aggregate results
echo "📊 Aggregating 153 experiment results..."
uv run python -m analysis.aggregate_results
echo "✓ Created: results/processed/aggregated.csv"
echo ""

# Generate paper content
echo "📝 Generating paper tables and figures..."
uv run python -m analysis.generate_tables
uv run python -m analysis.generate_figures
echo "✓ Created: paper/tables/*.tex and paper/figures/*.pdf"
echo ""

# Compile paper
echo "📄 Compiling paper PDF..."
cd paper && pdflatex main.tex && pdflatex main.tex && pdflatex main.tex && cd ..
echo "✓ Created: paper/main.pdf"
echo ""

echo "✅ Validation complete!"
echo ""
echo "Generated artifacts:"
echo "  - results/processed/aggregated.csv (aggregated results)"
echo "  - paper/tables/*.tex (12 LaTeX tables)"
echo "  - paper/figures/*.pdf (6 figures)"
echo "  - paper/main.pdf (28-page paper)"
echo ""
echo "To run full experiments (920 GPU-hours), see: EXPERIMENTS_REFERENCE.sh"

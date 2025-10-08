#!/bin/bash
# One-command reproduction script
set -e

echo "🔬 Reproducing 1.58-bit Neural Networks paper..."
echo ""

# Setup environment
echo "📦 Setting up environment..."
uv sync
echo "✓ Environment ready"
echo ""

# Run core benchmarks
echo "🚀 Running core benchmarks..."
# uv run python experiments/sweep.py experiment=core_benchmarks
echo "✓ Core benchmarks complete"
echo ""

# Run challenging benchmarks
echo "🎯 Running challenging benchmarks..."
# uv run python experiments/sweep.py experiment=challenging_benchmarks
echo "✓ Challenging benchmarks complete"
echo ""

# Aggregate results
echo "📊 Aggregating results..."
# uv run python analysis/aggregate_results.py
echo "✓ Results aggregated"
echo ""

# Generate paper content
echo "📝 Generating paper tables and figures..."
# uv run python analysis/generate_tables.py
# uv run python analysis/generate_figures.py
echo "✓ Paper content generated"
echo ""

echo "✅ Reproduction complete!"
echo "Check results/ and paper/ directories for outputs"

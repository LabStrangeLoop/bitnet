#!/bin/bash
# Check what content might be out of sync across venues

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PAPER_DIR"

VENUES=("tmlr" "neurips-workshop" "bmvc" "cvpr" "iclr" "neurips" "wacv")

echo "🔍 Checking for potentially out-of-sync content across venues"
echo ""

# Check if tables/figures were recently regenerated
if [ -d "../results/raw" ]; then
    latest_result=$(find ../results/raw -name "*.json" -type f -exec stat -f "%m %N" {} \; 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    latest_table=$(find tables -name "*.tex" -type f -exec stat -f "%m %N" {} \; 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

    if [ -n "$latest_result" ] && [ -n "$latest_table" ]; then
        result_time=$(stat -f "%m" "$latest_result" 2>/dev/null || echo "0")
        table_time=$(stat -f "%m" "$latest_table" 2>/dev/null || echo "0")

        if [ "$result_time" -gt "$table_time" ]; then
            echo "⚠️  WARNING: Results newer than tables"
            echo "   Latest result: $latest_result"
            echo "   Latest table: $latest_table"
            echo "   → Run: uv run python -m analysis.generate_tables"
            echo ""
        fi
    fi
fi

# Extract key numbers from each venue
echo "━━━ KEY NUMBERS BY VENUE ━━━"
echo ""

extract_number() {
    local venue=$1
    local pattern=$2
    local file="$venue/main.tex"

    if [ -f "$file" ]; then
        grep -o "$pattern" "$file" | head -1 || echo "N/A"
    else
        echo "N/A"
    fi
}

echo "Gap Recovery on CIFAR-10:"
for venue in "${VENUES[@]}"; do
    # Look for "88%" or "89%" in context of recovery
    recovery=$(grep -i "recover.*8[89]%" "$venue/main.tex" 2>/dev/null | head -1 || echo "N/A")
    printf "  %-20s %s\n" "$venue:" "$(echo $recovery | cut -c1-60)..."
done
echo ""

echo "CIFAR-100 'exceeds FP32' claim:"
for venue in "${VENUES[@]}"; do
    exceeds=$(grep -i "exceed.*FP32.*CIFAR-100\|CIFAR-100.*exceed.*FP32" "$venue/main.tex" 2>/dev/null | head -1 || echo "N/A")
    printf "  %-20s %s\n" "$venue:" "$(echo $exceeds | cut -c1-60)..."
done
echo ""

# Check abstract consistency
echo "━━━ ABSTRACT VERSIONS ━━━"
echo ""

for venue in "${VENUES[@]}"; do
    if [ -f "$venue/main.tex" ]; then
        abstract=$(sed -n '/\\begin{abstract}/,/\\end{abstract}/p' "$venue/main.tex" | wc -w)
        echo "$venue: $abstract words"
    fi
done
echo ""

# Check which venues mention ConvNeXt
echo "━━━ ARCHITECTURE COVERAGE ━━━"
echo ""

for venue in "${VENUES[@]}"; do
    if [ -f "$venue/main.tex" ]; then
        printf "  %-20s" "$venue:"
        grep -q "ConvNeXt" "$venue/main.tex" && printf " ✓ConvNeXt" || printf " ✗ConvNeXt"
        grep -q "MobileNetV2" "$venue/main.tex" && printf " ✓MobileNet" || printf " ✗MobileNet"
        grep -q "EfficientNet" "$venue/main.tex" && printf " ✓EfficientNet" || printf " ✗EfficientNet"
        echo ""
    fi
done
echo ""

# Show citation count
echo "━━━ BIBLIOGRAPHY SIZE ━━━"
echo ""

for venue in "${VENUES[@]}"; do
    if [ -f "$venue/main.tex" ]; then
        # Count bibitem entries in venue-specific bib or shared
        bib_file=$(grep -o "\\\\input{.*bibliography.*}" "$venue/main.tex" | sed 's/.*{\(.*\)}/\1/' | head -1)
        if [ -n "$bib_file" ]; then
            # Handle relative paths
            if [[ "$bib_file" == ../* ]]; then
                bib_path="$bib_file.tex"
            else
                bib_path="$venue/$bib_file.tex"
            fi

            if [ -f "$bib_path" ]; then
                count=$(grep -c "\\\\bibitem{" "$bib_path" 2>/dev/null || echo "0")
                printf "  %-20s %d citations\n" "$venue:" "$count"
            fi
        fi
    fi
done
echo ""

echo "━━━ RECOMMENDATIONS ━━━"
echo ""
echo "After running new experiments:"
echo "  1. uv run python -m analysis.aggregate_results"
echo "  2. uv run python -m analysis.generate_tables"
echo "  3. uv run python -m analysis.generate_figures"
echo "  4. Review this output again"
echo "  5. Update key numbers in abstracts manually"
echo ""
echo "To compare two venues:"
echo "  diff tmlr/main.tex neurips-workshop/main.tex | head -50"

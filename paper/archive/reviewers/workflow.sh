#!/bin/bash
# Automated paper review workflow

set -e

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PAPER_DIR"

VENUES=("tmlr" "neurips-workshop" "bmvc" "cvpr" "iclr" "neurips" "wacv")

# Detect clipboard command
if command -v pbcopy &> /dev/null; then
    CLIP_CMD="pbcopy"
elif command -v xclip &> /dev/null; then
    CLIP_CMD="xclip -selection clipboard"
elif command -v clip.exe &> /dev/null; then
    CLIP_CMD="clip.exe"
else
    CLIP_CMD=""
fi

show_help() {
    cat << EOF
📝 Paper Review Workflow (Streamlined)

MAIN COMMANDS:
    next <venue>               🚀 Start next review round (automated)
    analyze <venue>            📊 Analyze latest review (actions + plan + scores)
    summary <venue>            📋 Show current status

GRANULAR COMMANDS:
    build <venue|all>          Build paper(s)
    actions <venue>            Extract action items
    plan <venue>               Generate implementation plan
    scores <venue>             Show score progression
    status                     Show all venues build status
    clean                      Remove build artifacts

STREAMLINED WORKFLOW (3 steps):
    1. ./reviewers/workflow.sh next neurips-workshop
       → Builds paper, creates round dir, copies review prompt to clipboard

    2. Paste to Claude → get review → save to displayed path

    3. ./reviewers/workflow.sh analyze neurips-workshop
       → Shows actions, generates PLAN.md, displays scores

    4. Fix paper → repeat from step 1

TIPS:
    - 'next' auto-detects round number
    - 'analyze' does everything after you save the review
    - Use 'summary' to check current state anytime

EOF
}

get_next_round() {
    local venue=$1
    local latest=$(find reviewers -name "${venue//-/_}.md" 2>/dev/null | sort -r | head -1)
    if [ -z "$latest" ]; then
        echo "1"
    else
        local current_round=$(echo "$latest" | grep -oE 'round_[0-9]+' | grep -oE '[0-9]+')
        echo $((current_round + 1))
    fi
}

get_current_round() {
    local venue=$1
    local latest=$(find reviewers -name "${venue//-/_}.md" 2>/dev/null | sort -r | head -1)
    if [ -z "$latest" ]; then
        echo "0"
    else
        echo "$latest" | grep -oE 'round_[0-9]+' | grep -oE '[0-9]+'
    fi
}

build_venue() {
    local venue=$1
    echo "🔨 Building $venue..."
    cd "$venue" && make > /dev/null 2>&1 && echo "   ✓ Built successfully" || echo "   ✗ Build failed"
    cd ..
}

show_status() {
    echo "=== Paper Status ==="
    for venue in "${VENUES[@]}"; do
        [ -f "$venue/main.pdf" ] && echo "✓ $venue" || echo "  $venue"
    done
}

case "${1:-}" in
    next)
        venue=$2
        if [ -z "$venue" ]; then
            echo "Error: venue required"
            echo "Usage: ./reviewers/workflow.sh next <venue>"
            exit 1
        fi

        echo "🚀 Starting next review round for $venue"
        echo ""

        # Build paper
        build_venue "$venue"
        echo ""

        # Detect next round
        next_round=$(get_next_round "$venue")
        round_dir="reviewers/round_$next_round"
        mkdir -p "$round_dir"
        echo "📁 Created $round_dir/"
        echo ""

        # Generate save path
        save_path="$round_dir/${venue//-/_}.md"

        # Generate review prompt
        echo "📝 Generating review prompt (Round $next_round)..."
        review_output=$(mktemp)
        {
            echo "=== Reviewer Prompt for $venue (Round $next_round) ==="
            cat "reviewers/round_0/${venue//-/_}.md"
            echo ""
            echo "--- PASTE PAPER BELOW ---"
            echo ""
            pdftotext "$venue/main.pdf" - 2>/dev/null || cat "$venue/main.tex"
        } > "$review_output"

        # Copy to clipboard
        if [ -n "$CLIP_CMD" ]; then
            cat "$review_output" | $CLIP_CMD
            echo "   ✓ Copied to clipboard"
        else
            echo "   ⚠️  No clipboard tool found (install pbcopy/xclip)"
        fi

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "NEXT STEPS:"
        echo "1. Paste prompt to Claude"
        echo "2. Save review to: $save_path"
        echo "3. Run: ./reviewers/workflow.sh analyze $venue"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        rm -f "$review_output"
        ;;

    analyze)
        venue=$2
        if [ -z "$venue" ]; then
            echo "Error: venue required"
            echo "Usage: ./reviewers/workflow.sh analyze <venue>"
            exit 1
        fi

        echo "📊 Analyzing latest review for $venue"
        echo ""

        LATEST_REVIEW=$(find reviewers -name "${venue//-/_}.md" | sort -r | head -1)

        if [ ! -f "$LATEST_REVIEW" ]; then
            echo "❌ No review found for $venue"
            exit 1
        fi

        current_round=$(basename $(dirname "$LATEST_REVIEW"))
        echo "📍 Using: $LATEST_REVIEW"
        echo ""

        # Extract actions
        echo "━━━ ACTION ITEMS ━━━"
        echo ""
        echo "🔴 MAJOR Issues (Must Fix):"
        major=$(sed -n '/\[MAJOR\]/,/\[MINOR\]/p' "$LATEST_REVIEW" | grep -v '\[MINOR\]' | grep -E '^\s*-' || echo "")
        if [ -z "$major" ]; then
            echo "   ✓ None"
        else
            echo "$major" | sed 's/^/   /'
        fi
        echo ""

        echo "🟡 MINOR Issues (Should Fix):"
        minor=$(sed -n '/\[MINOR\]/,/Questions for Authors/p' "$LATEST_REVIEW" | grep -v 'Questions for Authors' | grep -E '^\s*-' || echo "")
        if [ -z "$minor" ]; then
            echo "   ✓ None"
        else
            echo "$minor" | sed 's/^/   /'
        fi
        echo ""

        echo "💡 Suggestions for Strengthening:"
        suggestions=$(sed -n '/Suggestions for Strengthening/,/^$/p' "$LATEST_REVIEW" | grep -E '^\s*[0-9]+\.' || echo "")
        if [ -z "$suggestions" ]; then
            echo "   None"
        else
            echo "$suggestions" | sed 's/^/   /'
        fi
        echo ""

        # Generate plan
        echo "━━━ IMPLEMENTATION PLAN ━━━"
        echo ""
        plan_file="PLAN.md"
        {
            echo "# Implementation Plan: $venue ($current_round)"
            echo ""
            echo "## Issues to Address"
            echo ""
            echo "### Critical (MAJOR)"
            if [ -z "$major" ]; then
                echo "✓ None"
            else
                echo "$major"
            fi
            echo ""
            echo "### Important (MINOR)"
            if [ -z "$minor" ]; then
                echo "✓ None"
            else
                echo "$minor"
            fi
            echo ""
            echo "## Enhancements"
            if [ -z "$suggestions" ]; then
                echo "None"
            else
                echo "$suggestions"
            fi
            echo ""
            echo "## Decision"
            grep -A 2 "^  Decision" "$LATEST_REVIEW" | tail -1 || echo "Not specified"
        } > "$plan_file"
        echo "   ✓ Saved to $plan_file"
        echo ""

        # Show scores
        echo "━━━ SCORE PROGRESSION ━━━"
        echo ""
        for round_dir in reviewers/round_*; do
            round=$(basename "$round_dir")
            review_file="$round_dir/${venue//-/_}.md"
            if [ -f "$review_file" ]; then
                echo "[$round]"
                grep "│ Overall" "$review_file" | head -1 | sed 's/^/   /' || \
                    grep "Overall.*[0-9]/10" "$review_file" | head -1 | sed 's/^/   /'
            fi
        done
        echo ""

        # Decision
        decision=$(grep -A 2 "^  Decision" "$LATEST_REVIEW" | tail -1 | xargs || echo "Unknown")
        echo "━━━ CURRENT DECISION: $decision ━━━"
        echo ""
        ;;

    summary)
        venue=$2
        if [ -z "$venue" ]; then
            echo "Error: venue required"
            echo "Usage: ./reviewers/workflow.sh summary <venue>"
            exit 1
        fi

        echo "📋 Summary: $venue"
        echo ""

        LATEST_REVIEW=$(find reviewers -name "${venue//-/_}.md" | sort -r | head -1)

        if [ ! -f "$LATEST_REVIEW" ]; then
            echo "❌ No reviews yet"
            echo ""
            echo "Start with: ./reviewers/workflow.sh next $venue"
            exit 0
        fi

        current_round=$(basename $(dirname "$LATEST_REVIEW"))
        echo "Round: $current_round"

        # Latest score
        latest_score=$(grep "│ Overall" "$LATEST_REVIEW" | head -1 | grep -oE '[0-9]+/10' || echo "N/A")
        echo "Score: $latest_score"

        # Decision
        decision=$(grep -A 2 "^  Decision" "$LATEST_REVIEW" | tail -1 | xargs || echo "Unknown")
        echo "Decision: $decision"

        # Count issues
        major_count=$(sed -n '/\[MAJOR\]/,/\[MINOR\]/p' "$LATEST_REVIEW" | grep -v '\[MINOR\]' | grep -cE '^\s*-' || echo "0")
        minor_count=$(sed -n '/\[MINOR\]/,/Questions for Authors/p' "$LATEST_REVIEW" | grep -v 'Questions for Authors' | grep -cE '^\s*-' || echo "0")
        echo "Issues: $major_count MAJOR, $minor_count MINOR"

        # Build status
        if [ -f "$venue/main.pdf" ]; then
            echo "Build: ✓ Up to date"
        else
            echo "Build: ✗ Not built"
        fi

        echo ""
        if [ "$decision" = "Accept" ] || [ "$decision" = "Strong Accept" ]; then
            echo "🎉 Ready for submission!"
        else
            echo "Next: ./reviewers/workflow.sh next $venue"
        fi
        ;;

    # Granular commands (kept for flexibility)
    build)
        [ "$2" = "all" ] && for v in "${VENUES[@]}"; do build_venue "$v"; done || build_venue "$2"
        ;;
    actions)
        echo "=== Action Items for $2 ==="
        echo ""
        LATEST_REVIEW=$(find reviewers -name "${2//-/_}.md" | sort -r | head -1)
        if [ -f "$LATEST_REVIEW" ]; then
            echo "From: $LATEST_REVIEW"
            echo ""
            echo "## MAJOR Issues (Must Fix)"
            sed -n '/\[MAJOR\]/,/\[MINOR\]/p' "$LATEST_REVIEW" | grep -v '\[MINOR\]' | grep -E '^\s*-' || echo "  None found"
            echo ""
            echo "## MINOR Issues (Should Fix)"
            sed -n '/\[MINOR\]/,/Questions for Authors/p' "$LATEST_REVIEW" | grep -v 'Questions for Authors' | grep -E '^\s*-' || echo "  None found"
            echo ""
            echo "## Suggestions for Strengthening"
            sed -n '/Suggestions for Strengthening/,/^$/p' "$LATEST_REVIEW" | grep -E '^\s*[0-9]+\.' || echo "  None found"
        else
            echo "No review found for $2"
        fi
        ;;
    plan)
        echo "=== Implementation Plan from Latest Review ==="
        echo ""
        LATEST_REVIEW=$(find reviewers -name "${2//-/_}.md" | sort -r | head -1)
        if [ -f "$LATEST_REVIEW" ]; then
            echo "# Implementation Plan: $2"
            echo ""
            echo "## Issues to Address"
            echo ""
            echo "### Critical (MAJOR)"
            sed -n '/\[MAJOR\]/,/\[MINOR\]/p' "$LATEST_REVIEW" | grep -v '\[MINOR\]' | grep -E '^\s*-' | sed 's/^  *//' || echo "None"
            echo ""
            echo "### Important (MINOR)"
            sed -n '/\[MINOR\]/,/Questions for Authors/p' "$LATEST_REVIEW" | grep -v 'Questions for Authors' | grep -E '^\s*-' | sed 's/^  *//' || echo "None"
            echo ""
            echo "## Enhancements"
            sed -n '/Suggestions for Strengthening/,/^$/p' "$LATEST_REVIEW" | grep -E '^\s*[0-9]+\.' | sed 's/^  *//' || echo "None"
            echo ""
            echo "## Decision"
            grep -A 2 "^  Decision" "$LATEST_REVIEW" | tail -1
        else
            echo "No review found for $2"
        fi
        ;;
    scores)
        echo "=== Score Progression for $2 ==="
        echo ""
        for round_dir in reviewers/round_*; do
            round=$(basename "$round_dir")
            review_file="$round_dir/${2//-/_}.md"
            if [ -f "$review_file" ]; then
                echo "[$round]"
                grep -A 5 "^│.*Score" "$review_file" | grep "^│" | grep -v "Criterion" | grep -v "^├" || \
                    grep "Overall.*[0-9]/10" "$review_file" | head -1
                echo ""
            fi
        done
        ;;
    status)
        show_status
        ;;
    clean)
        rm -f *.aux *.log *.out *.fls *.fdb_latexmk *.synctex.gz
        for v in "${VENUES[@]}"; do (cd "$v" && make clean 2>/dev/null); done
        ;;
    *)
        show_help
        ;;
esac

# Paper Compilation Guide

This directory contains the LaTeX source for the TMLR paper submission.

## Building the Paper

```bash
cd paper
make
```

This will:

1. Run `pdflatex` three times to resolve cross-references
2. Generate `main.pdf` (28 pages, ~550 KB)

## Clean Build

```bash
make clean  # Remove auxiliary files
make        # Rebuild from scratch
```

## Directory Structure

```
paper/
├── main.tex              # Main paper file
├── preamble.tex          # LaTeX packages and setup
├── macros.tex            # Custom commands
├── bibliography.tex      # References
├── sections/             # Paper sections
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methods_compressed.tex
│   ├── 04_architecture.tex
│   ├── 05_results.tex
│   ├── 06_discussion.tex
│   ├── 07_conclusion.tex
│   └── 08_appendix_reproducibility.tex
├── tables/               # LaTeX tables (generated)
├── figures/              # PDF figures (generated)
└── Makefile
```

## Regenerating Tables and Figures

All tables and figures are programmatically generated from experiment results:

```bash
# From project root
cd /Users/dariocazzani/code/lab-strange-loop/bitnet

# Aggregate experiment results
uv run python -m analysis.aggregate_results

# Generate LaTeX tables
uv run python -m analysis.generate_tables

# Generate PDF figures
uv run python -m analysis.generate_figures

# Rebuild paper
cd paper && make
```

## Requirements

- LaTeX distribution (texlive-full or equivalent)
- `pdflatex` command available in PATH
- All required packages specified in `preamble.tex`

## Troubleshooting

**Missing packages:**

```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS
brew install --cask mactex
```

**"File not found" errors:**

- Verify all `\input{}` paths are correct
- Check that `tables/` and `figures/` directories exist
- Ensure generated content is up to date (re-run analysis scripts)

**"Undefined references" or "??" in output:**

- Run `pdflatex` three times to resolve all cross-references
- Use `make` which automatically runs multiple passes

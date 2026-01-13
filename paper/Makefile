# Paper build commands

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean watch

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(MAIN)
	# $(BIBTEX) $(MAIN)  # Uncomment when using references.bib
	# $(LATEX) $(MAIN)
	# $(LATEX) $(MAIN)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

# Watch for changes (requires latexmk)
watch:
	latexmk -pdf -pvc $(MAIN)

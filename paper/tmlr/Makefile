MAIN = main
LATEX = pdflatex

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex preamble.tex macros.tex bibliography.tex
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

.PHONY: all clean

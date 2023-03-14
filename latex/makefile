
PDFLATEX= pdflatex
BIBTEX= bibtex


SRC := $(shell egrep -l '^[^%]*\\begin\{document\}' *.tex | sed -e 's/.tex//')


all :
	$(PDFLATEX) $(SRC) && $(BIBTEX) $(SRC) && $(PDFLATEX) $(SRC) && $(PDFLATEX) $(SRC)
#	$(PDFLATEX) $(SRC) && $(PDFLATEX) $(SRC)


clean :
	rm -f *.bbl *.aux *.blg *.log *.lof *.lot *.toc


PDFLATEX= pdflatex
BIBTEX= bibtex
GLOSSARIES= makeglossaries


SRC := $(shell egrep -l '^[^%]*\\begin\{document\}' *.tex | sed -e 's/.tex//')


all :
	$(PDFLATEX) $(SRC) && $(BIBTEX) $(SRC) && $(GLOSSARIES) $(SRC) && $(PDFLATEX) $(SRC) && $(PDFLATEX) $(SRC)
#	$(PDFLATEX) $(SRC) && $(PDFLATEX) $(SRC)


clean :
	rm -f *.acn *.aux *.bbl *.blg *.brf *glo *.glsdefs *.ist *.lof *.log *.lot *.out *.toc *gls *glg

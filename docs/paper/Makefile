# make        - build main.pdf (full pdflatex/bibtex/pdflatex/pdflatex pass)
# make clean  - remove intermediate build files, keep main.pdf
# make distclean - also remove main.pdf

MAIN := main

.PHONY: all clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(MAKE) clean

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).bbl $(MAIN).blg \
	      missfont.log build.log $(MAIN).synctex.gz $(MAIN).fdb_latexmk $(MAIN).fls

distclean: clean
	rm -f $(MAIN).pdf

main.pdf: main.tex bibliography.bib neurips.sty extra_pkgs.tex
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.synctex.gz

.PHONY: clean

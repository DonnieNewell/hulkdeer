paper: clean
	pdflatex PPoPP-report.tex
	pdflatex PPoPP-report.tex
	bibtex -min-crossrefs=100 PPoPP-report
	pdflatex PPoPP-report.tex
	pdflatex PPoPP-report.tex

clean: tidy
	rm -f PPoPP-report.pdf

tidy:
	rm -f *.log *.aux *.bbl *.blg

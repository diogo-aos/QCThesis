# ---------------------------------------------------------
# type "make" command in Unix to create ExtendedAbstract.pdf file 
# ---------------------------------------------------------

# Main filename
MAIN=ExtendedAbstract

# ---------------------------------------------------------

all:
	pdflatex ${MAIN}
	bibtex   ${MAIN}
	pdflatex ${MAIN}
	pdflatex ${MAIN}

clean:
	(rm -rf *.aux *.bbl *.blg *.log)

veryclean: clean
	(rm -rf *.pdf)

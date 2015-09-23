# ---------------------------------------------------------
# type "make" command in Unix to create the PDF file 
# ---------------------------------------------------------

# Main filename
FILE=Thesis
FOLDER_ALL=allchapters
FILE_ALL=Thesis_AllChapters

# ---------------------------------------------------------

all:
	pdflatex  -shell-escape ${FILE} 
	makeindex $(FILE).nlo -s nomencl.ist -o $(FILE).nls
	makeindex $(FILE).glo -s $(FILE).ist -o $(FILE).gls -t $(FILE).glg
	bibtex ${FILE}
	pdflatex  -shell-escape ${FILE}
	pdflatex  -shell-escape ${FILE}

chapters:
	pdflatex  -shell-escape ${FILE_ALL}
	makeindex $(FILE_ALL).nlo -s nomencl.ist -o $(FILE_ALL).nls
	makeindex $(FILE_ALL).glo -s $(FILE_ALL).ist -o $(FILE_ALL).gls -t $(FILE_ALL).glg
	bibtex ${FILE_ALL}
	pdflatex  -shell-escape ${FILE_ALL}
	pdflatex  -shell-escape ${FILE_ALL}

clean:
	(rm -rf *.aux *.bbl *.blg *.glg *.glo *.gls *.ilg *.ist *.lof *.log *.lot *.nlo *.nls *.out *.toc)

veryclean:
	make clean
	rm -f *~ *.*%
	rm -f $(FILE).pdf $(FILE).
	rm -f $(FILE_ALL).pdf $(FILE_ALL).


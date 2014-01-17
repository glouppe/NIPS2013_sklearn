SOURCE ?= poster
BUILD_DIR ?= .
PDFLATEX = pdflatex -shell-escape
BIBTEX ?= bibtex

all: build

build:
	@mkdir -p $(BUILD_DIR)
	$(PDFLATEX) $(SOURCE).tex
	# make bibtex
	$(PDFLATEX) $(SOURCE).tex
	$(PDFLATEX) $(SOURCE).tex
	@mv $(BUILD_DIR)/$(SOURCE).pdf $(SOURCE).pdf

bibtex:
	$(BIBTEX) $(BUILD_DIR)/$(SOURCE)

.PHONY: clean

clean:
	rm -f *.aux *.log *.nav *.snm *.vrb *.out *.bbl *.toc $(SOURCE).pdf


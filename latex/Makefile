##### Variables #############

# How many tries before giving up on resolution of references:
TRIES=7

# use the top one to make it automatically duplex.
DUPLEX=-h sty/duplex
#DUPLEX=

# Basename for result
TARGET=msc-sigurdur-2019
#BIBFILE=${TARGET}.bib
BIBFILE=references.bib
PROPTARGET=proposal
GRAPHICSDIR=graphics

# Sources
TEXSRC:=$(wildcard *.tex) $(wildcard *.cls) ${BIBFILE} $(wildcard ${GRAPHICSDIR}/*.latex) $(wildcard graphics/*)
EPSSRC:= $(wildcard ${GRAPHICSDIR}/*.eps) $(wildcard Numerical/*.eps)
PDFSRC:= $(wildcard ${GRAPHICSDIR}/*.pdf) $(wildcard Numerical/*.pdf)
TEXINPUTS:=".:./sty//:"

# Programs  -- put absolute paths here if you must
LATEX:=TEXINPUTS=${TEXINPUTS}; export TEXINPUTS; latex
PDFLATEX:=TEXINPUTS=${TEXINPUTS}; export TEXINPUTS; latexmk -pdf
BIBPROG:=bibtex  # can also be bibtex or bibtex8

# ATTENTION!
# File-extensions to delete recursive from here
EXTENSION=acn acr alg aux bbl bcf blg glo glg gls glsdefs idx ind ilg ist lof log lol lot lox mw out run.xml toc sbl slg sym

#############################

######################################
######################################
##### Targets for main ###############
######################################

all: pdf cleanExtra

pdf:
	${PDFLATEX} ${TARGET}
	# ${BIBPROG}  ${TARGET}
	# ${PDFLATEX} ${TARGET}
	# ${PDFLATEX} ${TARGET}

# HTML
html: ${TEXSRC} ${PDFSRC}
	latex ${TARGET}.tex
	bibtex ${TARGET}
	latex ${TARGET}.tex
	latex2html \
	-short_index -split 3 \
	-dir www -numbered_footnotes -no_footnode \
	-antialias -html_version 4.0 \
	${TARGET}.tex

biblio:
	echo "Running ${BIBPROG} on ${TARGET}"
	${BIBPROG} ${TARGET}

viewer:
	evince ${TARGET}.pdf &

# Clean
clean:
	for EXT in ${EXTENSION}; \
	do \
	find `pwd` -name \*\.$${EXT} -exec rm -v \{\} \; ;\
	done
	rm -f ${TARGET}.{dvi,pdf*,ps*}
	rm -f www/*\.*
	rm -f *-blx.bib

cleanExtra:
	for EXT in ${EXTENSION}; \
	do \
	find `pwd` -name \*\.$${EXT} -exec rm -v \{\} \; ;\
	done
	rm -rf ${TARGET}.{dvi,ps*}
	rm -f www/*\.*
	rm -f *-blx.bib

.SUFFIXES: .pdf .eps .dot .dvi .tex .ps

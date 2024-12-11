#!/usr/bin/env bash

# Nombre del archivo principal (sin la extensión .tex)
MAIN="/home/tonix/Documents/PhdDegreeCode/Documents/MetodologiaProjectoFinal/LatexProject/main"

rm *.pdf
rm *.aux *.bbl *.bcf *.blg *.log *.toc
# Primero compilamos con pdflatex
pdflatex "$MAIN.tex"

# Compilamos con biber (para procesar la bibliografía)
biber "$MAIN"

# Compilamos dos veces más con pdflatex para actualizar referencias
pdflatex "$MAIN.tex"
pdflatex "$MAIN.tex"

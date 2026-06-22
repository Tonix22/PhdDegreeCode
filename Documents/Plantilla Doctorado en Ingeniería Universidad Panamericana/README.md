# Plantilla Doctorado en Ingeniería — Universidad Panamericana

## Compilación

El documento principal es `main.tex`. En macOS o Linux, desde esta carpeta:

```sh
./compile.sh
```

El resultado se guarda en `main.pdf`. El script utiliza `latexmk`, por lo que
ejecuta automáticamente las pasadas necesarias de pdfLaTeX y BibTeX para
resolver el índice, las referencias cruzadas y la bibliografía.

Opciones disponibles:

```sh
./compile.sh --rebuild  # limpia auxiliares y recompila todo
./compile.sh --clean    # elimina solamente archivos auxiliares
./compile.sh --watch    # recompila al detectar cambios (Ctrl+C para salir)
./compile.sh --help     # muestra la ayuda
```

## Requisitos

Se necesitan `latexmk`, `pdflatex`, `bibtex` y los paquetes LaTeX utilizados en
`Config/packages.tex`. La opción más simple es instalar una distribución TeX
completa:

### macOS

Con [Homebrew](https://brew.sh/):

```sh
brew install --cask mactex-no-gui
```

Después de la instalación, abre una terminal nueva. El script también agrega
automáticamente `/Library/TeX/texbin` al `PATH` cuando existe.

### Ubuntu o Debian

```sh
sudo apt update
sudo apt install latexmk texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

### Fedora

```sh
sudo dnf install latexmk texlive-scheme-full
```

En Windows se puede ejecutar `compile.sh` desde WSL con los paquetes de
Ubuntu/Debian anteriores. Alternativamente, se puede instalar MiKTeX y compilar
`main.tex` con `latexmk` desde su propia consola.

## Compilación manual

El comando equivalente al script es:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

Los detalles de cualquier error quedan en `main.log`.

#!/usr/bin/env bash

set -Eeuo pipefail

# Compila siempre desde la carpeta donde se encuentra este script. Esto permite
# ejecutarlo desde cualquier directorio y también admite espacios en la ruta.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# MacTeX no siempre queda en PATH cuando el script se ejecuta desde un editor.
if [[ -d /Library/TeX/texbin ]]; then
  PATH="/Library/TeX/texbin:$PATH"
fi

print_usage() {
  cat <<'EOF'
Uso: ./compile.sh [opcion]

Sin opcion       Compila main.tex y genera main.pdf.
--rebuild, -r    Borra auxiliares y recompila todo.
--clean, -c      Borra solamente archivos auxiliares.
--watch, -w      Recompila cada vez que cambia un archivo fuente.
--help, -h       Muestra esta ayuda.
EOF
}

installation_help() {
  cat >&2 <<'EOF'

No se encontro una instalacion de LaTeX con latexmk y pdflatex.

Instalacion recomendada:
  macOS (Homebrew): brew install --cask mactex-no-gui
  Ubuntu/Debian:    sudo apt install latexmk texlive-latex-base texlive-latex-extra texlive-fonts-recommended
  Fedora:           sudo dnf install latexmk texlive-scheme-full

En macOS, despues de instalar MacTeX, abre una terminal nueva o agrega
/Library/TeX/texbin al PATH. En Windows, ejecuta este script desde WSL.
EOF
}

if [[ ! -f main.tex ]]; then
  printf 'Error: no se encontro %s/main.tex\n' "$PROJECT_DIR" >&2
  exit 1
fi

missing=()
for command_name in latexmk pdflatex bibtex; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    missing+=("$command_name")
  fi
done

if (( ${#missing[@]} > 0 )); then
  printf 'Faltan los comandos requeridos: %s\n' "${missing[*]}" >&2
  installation_help
  exit 127
fi

mode="${1:-compile}"
case "$mode" in
  compile)
    ;;
  --rebuild|-r)
    latexmk -C main.tex
    ;;
  --clean|-c)
    latexmk -c main.tex
    printf 'Archivos auxiliares eliminados.\n'
    exit 0
    ;;
  --watch|-w)
    printf 'Vigilando cambios; presiona Ctrl+C para terminar.\n'
    exec latexmk -pdf -pvc -interaction=nonstopmode -halt-on-error \
      -file-line-error main.tex
    ;;
  --help|-h)
    print_usage
    exit 0
    ;;
  *)
    printf 'Opcion desconocida: %s\n\n' "$mode" >&2
    print_usage >&2
    exit 2
    ;;
esac

printf 'Compilando %s/main.tex...\n' "$PROJECT_DIR"
if latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -file-line-error main.tex; then
  printf '\nCompilacion terminada: %s/main.pdf\n' "$PROJECT_DIR"
else
  status=$?
  printf '\nLa compilacion fallo. Revisa main.log; los primeros errores suelen aparecer con "!" o con una ruta y numero de linea.\n' >&2
  exit "$status"
fi

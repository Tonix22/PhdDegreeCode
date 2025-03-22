#!/bin/bash
# Script: create_readmes_recursive.sh
# Este script realiza lo siguiente:
# 1. Crea archivos README.md en cada directorio (no oculto) hasta dos niveles de profundidad.
# 2. Genera un README.md principal en el directorio actual que lista los directorios de nivel 1.
# 3. En cada directorio de nivel 1, se agrega una sección "Subdirectories" con enlaces a sus subdirectorios (nivel 2).

# Paso 1: Crear README.md en cada directorio (hasta 2 niveles, sin incluir los ocultos)
find . -mindepth 1 -maxdepth 2 -type d ! -name ".*" | while read -r dir; do
    readme_path="$dir/README.md"
    if [ ! -f "$readme_path" ]; then
        echo "# README for $(basename "$dir")" > "$readme_path"
        echo "This is the README file for the directory: $(basename "$dir")." >> "$readme_path"
    fi
done

# Paso 2: Crear el README.md principal en el directorio actual (nivel 0)
main_readme="./README.md"
echo "# Main README" > "$main_readme"
echo "This file links to all the subdirectory README files:" >> "$main_readme"
echo "" >> "$main_readme"

# Listar directorios de nivel 1 (subdirectorios directos)
find . -mindepth 1 -maxdepth 1 -type d ! -name ".*" | while read -r dir; do
    dir_name=$(basename "$dir")
    echo "- [$dir_name]($dir/README.md)" >> "$main_readme"
done

# Paso 3: Para cada directorio de nivel 1, agregar en su README una lista de links a sus subdirectorios (nivel 2)
find . -mindepth 1 -maxdepth 1 -type d ! -name ".*" | while read -r dir; do
    # Buscar subdirectorios directos de este directorio
    subdirs=$(find "$dir" -mindepth 1 -maxdepth 1 -type d ! -name ".*")
    if [ -n "$subdirs" ]; then
        readme_path="$dir/README.md"
        # Agregar una sección si aún no existe
        echo "" >> "$readme_path"
        echo "## Subdirectories" >> "$readme_path"
        echo "" >> "$readme_path"
        find "$dir" -mindepth 1 -maxdepth 1 -type d ! -name ".*" | while read -r subdir; do
            subdir_name=$(basename "$subdir")
            echo "- [$subdir_name]($subdir/README.md)" >> "$readme_path"
        done
    fi
done

echo "README files have been created and updated recursively."

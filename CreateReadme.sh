#!/bin/bash

# Create README.md files in the first three levels of subdirectories
find . -mindepth 1 -maxdepth 3 -type d | while read -r dir; do
    readme_path="$dir/README.md"
    if [ ! -f "$readme_path" ]; then
        echo "# README for $(basename "$dir")" > "$readme_path"
        echo "This is the README file for the directory: $(basename "$dir")." >> "$readme_path"
    fi
done

# Create the main README.md in the current directory
main_readme="./README.md"
echo "# Main README" > "$main_readme"
echo "This file links to all the subdirectory README files:" >> "$main_readme"
echo "" >> "$main_readme"

# Add links to subdirectories in the main README.md
find . -mindepth 1 -maxdepth 3 -type d | while read -r dir; do
    dir_name=$(basename "$dir")
    echo "- [$dir_name]($dir/README.md)" >> "$main_readme"
done

echo "README files have been created and the main README has been updated."

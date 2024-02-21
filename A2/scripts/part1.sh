#!/bin/bash

# Check if the directory path is provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

directory_path=$1

# Check if the directory exists
if [ ! -d "$directory_path" ]; then
    echo "Error: Directory '$directory_path' not found."
    exit 1
fi

# Iterate through each file in the directory and pass it to main.py
for dir in "$directory_path"/*; do
    if [ -d "$dir" ]; then
        python3 src/main.py 1 "$dir" output/"$(basename "$dir")".png 
    fi
done

#!/bin/bash

# Check for the number of command-line arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_folder> <output_folder>"
  exit 1
fi

# Set the input and output folder paths from command-line arguments
input_folder="$1"
output_folder="$2"

# Ensure the input folder exists
if [ ! -d "$input_folder" ]; then
  echo "Input folder does not exist: $input_folder"
  exit 1
fi

# Create the output folder if it doesn't exist
if [ ! -d "$output_folder" ]; then
  mkdir -p "$output_folder"
fi

# Loop through files in the input folder
for file in "$input_folder"/*; do
  if [ -f "$file" ]; then
    # Extract the base name of the input file (without extension)
    base_name=$(basename "$file" .wav)

    # Construct the expected output file name
    output_file="${output_folder}/${base_name}_ola.wav"

    # Check if the output file already exists
    if [ ! -f "$output_file" ]; then
      # Run the Python command with the complete file path (including extension)
      python predict.py dset=chopin-11-44-one experiment=aeromamba +filename="$file" +output="$output_folder"
    else
      echo "Output file already exists: $output_file"
    fi
  fi
done

#!/bin/bash

# Check if the input directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <input_directory>"
  exit 1
fi

input_dir="$1"

# Check if the input directory exists
if [ ! -d "$input_dir" ]; then
  echo "Input directory does not exist: $input_dir"
  exit 1
fi

# Loop through all MP3 files in the input directory
for input_file in "$input_dir"/*.mp3; do
  # Check if there are no MP3 files in the directory
  if [ ! -f "$input_file" ]; then
    echo "No MP3 files found in the directory: $input_dir"
    exit 1
  fi

  # Extract the file name without the extension
  base_name=$(basename "$input_file" .mp3)

  # Create the output file name with WAV extension
  output_file="${input_dir}/${base_name}.wav"

  # Convert MP3 to WAV using ffmpeg
  ffmpeg -i "$input_file" -ac 1 -y "$output_file"

  # Check if the conversion was successful
  if [ $? -eq 0 ]; then
    # Remove the original MP3 file
    rm "$input_file"
    echo "Converted and replaced: $input_file"
  else
    echo "Error converting file: $input_file"
  fi
done

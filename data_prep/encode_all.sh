#!/bin/bash

# Check if the input directory, output directory, and bitrate are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: $0 <input_directory> <output_directory> <bitrate_kbps>"
  exit 1
fi

input_dir="$1"
output_dir="$2"
bitrate="$3"

# Check if the input directory exists
if [ ! -d "$input_dir" ]; then
  echo "Input directory does not exist: $input_dir"
  exit 1
fi

# Check if the output directory exists; if not, create it
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

# Loop through all WAV files in the input directory
for input_file in "$input_dir"/*.wav; do
  # Check if there are no WAV files in the directory
  if [ ! -f "$input_file" ]; then
    echo "No WAV files found in the directory: $input_dir"
    exit 1
  fi

  # Extract the file name without the extension
  base_name=$(basename "$input_file" .wav)

  # Create the output file name
  output_file="${output_dir}/${base_name}.mp3"

  # Convert WAV to MP3 at specified bitrate using ffmpeg
  ffmpeg -i "$input_file" -ab "${bitrate}k" -y "$output_file"

  # Check if the command was successful
  if [ $? -eq 0 ]; then
    echo "Encoded file created: $output_file"
  else
    echo "Error encoding file: $input_file"
  fi
done

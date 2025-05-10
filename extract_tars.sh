#!/bin/bash

# Set the path to your text file containing the tar file paths.
TEXT_FILE=${1}  # "path/to/your/file.txt"

SRC_DIR=${2}

# Set the destination directory where you want to extract the files.
DESTINATION_DIR=${3}  # "path/to/your/destination/directory"

# Create the destination directory if it doesn't exist.
mkdir -p "$DESTINATION_DIR"

# Loop through each line in the text file.
while IFS= read -r TAR_FILE; do

  TAR_FILE=${SRC_DIR}/${TAR_FILE}
  # Check if the tar file exists
  if [[ -f "$TAR_FILE" ]]; then
    # Extract the tar file into the destination directory.
    # The -xvf options are commonly used:
    #   -x: extract
    #   -v: verbose (shows the files being extracted)
    #   -f: specify the archive file
    # You might need to adjust these based on the archive type (e.g., .tar.gz, .tar.bz2, .tar.xz)
    # See below for handling different archive types.

    # Default for .tar files:
    tar -xf "$TAR_FILE" -C "$DESTINATION_DIR"

    # Example for .tar.gz files:
    # tar -xvzf "$TAR_FILE" -C "$DESTINATION_DIR"

    # Example for .tar.bz2 files:
    # tar -xvjf "$TAR_FILE" -C "$DESTINATION_DIR"

    # Example for .tar.xz files:
    # tar -xvJf "$TAR_FILE" -C "$DESTINATION_DIR"

    echo "Extracted: $TAR_FILE to $DESTINATION_DIR"
  else
    echo "Error: Tar file not found: $TAR_FILE"
  fi

done < "$TEXT_FILE"

echo "Extraction complete."
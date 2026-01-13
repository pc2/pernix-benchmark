#!/usr/bin/env bash

print_help() {
  cat <<'USAGE'
Usage: transformResults.sh [options] <input_file> <output_file>
USAGE
}

# Check dependencies installed: python3
if ! command -v python3 &> /dev/null; then
  echo "Error: 'python3' is not installed. Please install it to proceed."
  exit 1
fi

PARSED=$(getopt -o h --long help -n 'transformResults.sh' -- "$@") || { echo "Invalid options"; exit 2; }
eval set -- "$PARSED"
while true; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Invalid option: $1"
      exit 2
      ;;
  esac
done

if [ "$#" -ne 2 ]; then
  echo "Error: Exactly two positional arguments required."
  print_help
  exit 2
fi

INPUT_FILE="$1"
OUTPUT_FOLDER="$2"

if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Input file '$INPUT_FILE' does not exist."
  exit 1
fi

# check if input is tar.gz
if [[ ! "$INPUT_FILE" == *.tar.gz ]]; then
  echo "Error: Input file '$INPUT_FILE' is not a .tar.gz file."
  exit 1
fi

if [ ! -d "$OUTPUT_FOLDER" ]; then
  echo "Error: Output folder '$OUTPUT_FOLDER' does not exist."
  exit 1
fi

TMP_DIR=$(mktemp -d)
# trap 'rm -rf "$TMP_DIR"' EXIT
tar -xzf "$INPUT_FILE" -C "$TMP_DIR"
echo "$TMP_DIR"

shopt -s nullglob
files=("$TMP_DIR"/benchmark_*.json)
if [ "${#files[@]}" -eq 0 ]; then
  echo "Error: No bench_*.json files found in $TMP_DIR."
  exit 1
fi

setup_python_environment() {
  if ! command -v python3 &> /dev/null; then
    echo "python3 could not be found, please install it."
    exit 1
  fi

  if ! python3 -m venv .venv; then
    echo "Failed to create Python virtual environment."
    exit 1
  fi

  # shellcheck disable=SC1091
  source .venv/bin/activate
  trap 'deactivate' EXIT

  if ! pip install --upgrade pip; then
    echo "Failed to upgrade pip."
    exit 1
  fi

  if ! pip install pandas; then
    echo "Failed to install required Python packages."
    exit 1
  fi
}

setup_python_environment

for f in "${files[@]}"; do
  name=$(basename "$f")
  if [[ $name =~ ^benchmark(_pernix)?_(.+)_results\.json$ ]]; then
    id="${BASH_REMATCH[2]}"
  else
    echo "Skipping $name: unexpected filename format" >&2
    continue
  fi
  python3 internal/transformResults.py "$id" "$f" "$OUTPUT_FOLDER"
done
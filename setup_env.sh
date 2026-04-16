#!/bin/bash

ENV_NAME="gnn_mpro"
ENV_FILE="environment.yml"
MINICONDA_PATH="$HOME/miniconda3"

echo "Starting setup..."

# Miniconda setup
if ! command -v conda &> /dev/null; then
    if [ -f "$MINICONDA_PATH/bin/conda" ]; then
        echo "[*] Miniconda found. Starting..."
        source "$MINICONDA_PATH/bin/activate"
    else
        echo "[!] Miniconda not found."
        echo "[*] Downloading miniconda."
        
        curl -sO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        
        bash Miniconda3-latest-Linux-x86_64.sh -b -u -p "$MINICONDA_PATH"
        
        rm -f Miniconda3-latest-Linux-x86_64.sh
        
        "$MINICONDA_PATH/bin/conda" init bash &> /dev/null
        
        source "$MINICONDA_PATH/bin/activate"
        
        echo "[+] Installed miniconda."
    fi
fi

# Environment setup
if [ ! -f "$ENV_FILE" ]; then
    echo "[!] ERROR: $ENV_FILE not found."
    exit 1
fi

if conda env list | grep -q "^$ENV_NAME "; then
    echo "[*] Found '$ENV_NAME'. Verifying..."
    conda env update --name "$ENV_NAME" --file "$ENV_FILE" --prune -q
else
    echo "[*] Creating '$ENV_NAME'..."
    conda env create -f "$ENV_FILE" -q
fi

echo "Completed setup successfully"
echo "WARNING: If this environment has never been setup before, restart the terminal to apply changes."
echo "To start: conda activate $ENV_NAME"
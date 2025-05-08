#!/bin/bash

echo "Downloading PTB-XL dataset..."
# Using wget directly as it's a system command, not a Python dependency
wget -r -N -c -np --quiet --show-progress -P data/ https://physionet.org/files/ptb-xl/1.0.3/
echo "Download complete!"

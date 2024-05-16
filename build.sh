#!/bin/bash
# Change directory to folder containing the script
cd $(dirname $0)

# Install build tools
pip install build 

# Build the package
python3 -m build

# Install the latest local version of the package to current enviroment
file=$(ls dist/ -ABrt1 --group-directories-first | tail -n1) # find latest package file = newest file in dist folder
echo "Installing dist/$file"
pip3 install -v --force-reinstall "dist/$file"
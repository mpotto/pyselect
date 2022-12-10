#!/bin/bash

# Download datasets 
python scripts/data/download.py -d cpusmall
python scripts/data/download.py -d ailerons
python scripts/data/download.py -d amazon
python scripts/data/download.py -d higgs

# Process all datasets
python scripts/data/process.py -d cpusmall
python scripts/data/process.py -d ailerons
python scripts/data/download.py -d amazon
python scripts/data/download.py -d higgs


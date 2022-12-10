#!/bin/bash

# Download datasets 
python newv/data/download.py -d cpusmall
python newv/data/download.py -d ailerons

# Generate synthetic datasets
python newv/data/synthesize.py -d gse1 -n 1000 -s 0
python newv/data/synthesize.py -d gse1 -n 5000 -s 0
python newv/data/synthesize.py -d gse1 -n 10000 -s 0
python newv/data/synthesize.py -d gse1 -n 50000 -s 0

# Process all datasets
python newv/data/process.py -d cpusmall
python newv/data/process.py -d ailerons


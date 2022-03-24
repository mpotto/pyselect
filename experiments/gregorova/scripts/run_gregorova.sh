#!/bin/bash
poetry shell

python gregorova.py --multirun dataset.name=se1 dataset.train_size=1000,5000,10000,50000
python gregorova.py dataset.name=se2 dataset.train_size=1000,5000,10000,50000
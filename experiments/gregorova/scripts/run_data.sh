#!/bin/bash
poetry shell

python data.py --multirun name=se1 train_size=2000,6000,11000,51000
python data.py --multirun name=se2 train_size=2000,6000,11000,51000
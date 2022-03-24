#!/bin/bash
poetry shell

python rffnet.py --multirun dataset.name=se1 model.in_features=18 dataset.train_size=1000,5000,10000,50000
python rffnet.py --multirun dataset.name=se2 model.in_features=100 dataset.train_size=1000,5000,10000,50000

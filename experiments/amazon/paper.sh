#!/bin/bash
python experiments/higgs/prepare.py

mprof run experiments/amazon/logistic-l1.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/logistic-l1/amazon/metrics/memory.txt

mprof run experiments/amazon/logistic.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/logistic/amazon/metrics/memory.txt

mprof run experiments/amazon/rffnet.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/rffnet/amazon/metrics/memory.txt

mprof run experiments/amazon/xgb.py
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/xgb/amazon/metrics/memory.txt

mprof clean


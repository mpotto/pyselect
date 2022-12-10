#!/bin/bash
python experiments/higgs/prepare.py

mprof run experiments/higgs/logistic-l1.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/logistic-l1/higgs/metrics/memory.txt

mprof run experiments/higgs/logistic.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/logistic/higgs/metrics/memory.txt

mprof run experiments/higgs/rffnet.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/rffnet/higgs/metrics/memory.txt

mprof run experiments/higgs/xgb.py
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/xgb/higgs/metrics/memory.txt

mprof clean
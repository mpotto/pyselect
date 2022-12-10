#!/bin/bash
mprof run experiments/gregorova/se2/ard.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/ard/gregorova/se2/metrics/memory.txt

mprof run experiments/gregorova/se2/krr.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/krr/gregorova/se2/metrics/memory.txt

mprof run experiments/gregorova/se2/rffnet.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/rffnet/gregorova/se2/metrics/memory.txt

mprof run experiments/gregorova/se2/srff.py
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/srff/gregorova/se2/metrics/memory.txt

mprof clean

python experiments/gregorova/se2/rffnet_scaling.py

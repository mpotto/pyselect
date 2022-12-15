#!/bin/bash
mprof run experiments/gregorova/se1/ard.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/ard/se1/metrics/memory.txt

mprof run experiments/gregorova/se1/krr.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/krr/se1/metrics/memory.txt

mprof run experiments/gregorova/se1/rffnet.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/rffnet/se1/metrics/memory.txt

mprof run experiments/gregorova/se1/srff.py
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/srff/se1/metrics/memory.txt

mprof clean

python experiments/gregorova/se1/rffnet_scaling.py

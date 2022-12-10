#!/bin/bash
mprof run experiments/ailerons/ard.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/ard/ailerons/metrics/memory.txt

mprof run experiments/ailerons/krr.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/krr/ailerons/metrics/memory.txt

mprof run experiments/ailerons/rffnet.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/rffnet/ailerons/metrics/memory.txt

mprof run experiments/ailerons/srff.py
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/srff/ailerons/metrics/memory.txt

mprof clean


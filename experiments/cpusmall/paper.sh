#!/bin/bash
mprof run experiments/cpusmall/ard.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/ard/cpusmall/metrics/memory.txt

mprof run experiments/cpusmall/krr.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/krr/cpusmall/metrics/memory.txt

mprof run experiments/cpusmall/rffnet.py 
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/rffnet/cpusmall/metrics/memory.txt

mprof run experiments/cpusmall/srff.py
mprof peak | grep -Eo '[+-]?([0-9]*[.])?[0-9]+ MiB' > eval/benchmarks/srff/cpusmall/metrics/memory.txt

mprof clean


#!/bin/bash

declare -a subdirs=("data" "eval")
for subdir in "${subdirs[@]}"
do
    # Make directory creating parent directories if necessary
    cmd="mkdir -p $subdir"
    echo "$cmd"; $cmd
done

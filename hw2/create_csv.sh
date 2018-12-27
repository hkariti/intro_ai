#!/bin/bash
PYTHON_BIN=~/anaconda3/bin/python
ghosts=2
player=$1
n=$2
d_start=$3
d_end=$4
if [ -z "$d_start" ]; then
    d_start=2
fi
if [ -z "$d_end" ]; then
    d_end=4
fi

if [ -z "$player" -o -z "$n" ]; then
    echo "Usage: $0 PLAYER NUM_OF_GAMES"
    echo "Will run PLAYER on all layouts with depths: 2 3 4 for NUM games each"
    exit 1
fi
for d in `seq $d_start $d_end`; do
    for layout in `ls layouts|sed 's/.lay$//'`; do
        $PYTHON_BIN pacman.py -p $player -k $ghosts -n $n -a depth=$d -l $layout -q |
            awk "/Average turn time/ {turn_time+=\$NF} /Average Score/ {avag_score=\$NF} END {print \"$player,$d,$layout,\" avag_score \",\" turn_time/$n}"
    done
done

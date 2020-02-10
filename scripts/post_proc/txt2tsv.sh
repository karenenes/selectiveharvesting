#!/bin/bash

##
# E.g.:  for i in p2r_results/config2_binary/rr_max_*/extracted/*.txt; do ./txt2tsv.sh $i; done
##

FILENAME=$1
OUTFILE=${FILENAME%.txt}.tsv

grep -A2 'turn' $FILENAME | sed -n '3~4p' | awk '{print $1, $3}' > $OUTFILE
echo $OUTFILE

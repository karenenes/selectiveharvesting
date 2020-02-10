#!/bin/bash

SCRIPTDIR="$(dirname "$0")"


DIRECTORIES=$@

for DIRNAME in ${DIRECTORIES}; do
	for dataset in donors dbpedia citeseer wikipedia kickstarter; do
        	OUTPATH=${DIRNAME}/extracted/
	        echo "------> $dataset $DIRNAME"
	        if [[ ! -d  $OUTPATH ]]; then
	            mkdir $OUTPATH
	        fi
	    Rscript  $SCRIPTDIR/aggregate.R  $OUTPATH  ${DIRNAME}/${dataset}_*RData
        sh $SCRIPTDIR/txt2tsv.sh ${DIRNAME}/extracted/${dataset}*txt
  done
done

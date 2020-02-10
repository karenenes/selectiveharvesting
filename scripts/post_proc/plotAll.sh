#!/bin/bash

# Make sure env var D3TSDIR is properly set
if [ x$D3TSDIR == x ]; then
    echo "ERROR: environment variable D3TSDIR not defined."
    exit 1
elif [ ! -f $D3TSDIR/src/mab/main.R ]; then
    echo "ERROR: src/mab/main.R not found. $D3TSDIR is not a valid directory."
    exit 2
fi

# read arguments
if [ $# -ne 1 ]; then
    echo "ERROR: Incorrect number of arguments."
    echo "Usage:"
    echo "$0 <results dir>"
    exit 0
else
    RESULTSDIR="$(pwd)/$1"
    if [ ! -d $RESULTSDIR ]; then
        echo "ERROR: $RESULTSDIR is not a valid directory."
    fi
fi

# go to the scripts folder
cd "$(dirname "$0")"

# call plotScript
Rscript plotStandalone.R $RESULTSDIR

# return to previous folder
cd -

echo ""
echo "RESULTS WERE SAVED IN FOLDER $RESULTSDIR"

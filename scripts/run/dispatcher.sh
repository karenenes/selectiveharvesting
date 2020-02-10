#!/bin/bash

# Make sure env var D3TSDIR is properly set
if [[ x$D3TSDIR == x ]]; then
    echo "ERROR: environment variable D3TSDIR not defined."
    exit 1
elif [[ ! -f $D3TSDIR/src/mab/main.R ]]; then
    echo "ERROR: $D3TSDIR is not a valid directory."
    exit 2
fi

# simulation parameters
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

# cluster parameters
NODES=1
PPN=20
WALLTIME="4:00:00"
MEM=200gb


DATASET=$1
QUEUENAME=$2

MSC="dts.5"
MSC="rr"
SCOREMAPPER="geom_dyn"
SCOREMAPPER="max"
FEATURESET="only_structural"
FEATURESET="all"
SETTINGS="adaboost.pickOne"
SETTINGS="adaboost"
SETTINGS="bagging.pickOne"
SETTINGS="bagging_mf100"
SETTINGS="pyboosting_nWL100"
SETTINGS="randomforest.rf"
SETTINGS="bagging_mf100_ms10_md5.pickOne"
SETTINGS="config2.p"
SETTINGS="config2"
SETTINGS="activesearch"
SETTINGS="mod"
SETTINGS="listnet"
SETTINGS="randomforest.p"
SETTINGS="svm"
NTESTS=100

# dataset specific settings
MAXTESTS=1000
case $DATASET in
    dbpedia*)
        MAXTESTS=725
        WALLTIME="1:15:00"
        MEMPERTSK=3
	;;
    citeseer*)
        MAXTESTS=1583
        WALLTIME="1:15:00"
        MEMPERTSK=3
	;;
    wikipedia*)
        MAXTESTS=202
        WALLTIME="1:45:00"
        MEMPERTSK=3
	;;
    blogcatalog)
        MAXTESTS=986
        MEMPERTSK=3
	;;
    donors)
        MAXTESTS=56
        WALLTIME="0:15:00"
        MEMPERTSK=1
	;;
    donors*)
        MAXTESTS=48
        WALLTIME="0:15:00"
        MEMPERTSK=1
	;;
    dblp)
        MAXTESTS=7556
        MEMPERTSK=5
	;;
    lj)
        MAXTESTS=1441
        MEMPERTSK=10
	;;
    kickstarter*)
        MAXTESTS=1457
        MEMPERTSK=3
	;;
    *)
        echo "Invalid dataset $1"
        quit
        ;;
esac

if [[ $NTESTS -gt $MAXTESTS ]]; then
    echo "Overriding NTESTS; now set to $MAXTESTS"
    NTESTS=$MAXTESTS
fi

# ribeirob specific settings
WALLTIME=4:00:00
if [[ $QUEUENAME == "ribeirob" ]]; then
    NODES=1
    PPN=20
    WALLTIME="60:00:00"
elif [[ $QUEUENAME == "debug" ]]; then
    WALLTIME="00:30:00"
fi

# "acconte" for the fact that conte has only 16 cores 
HOSTNAME=`hostname | cut -d- -f1`
if [[ $HOSTNAME == "conte" ]]; then
    case $DATASET in
        dblp)
            PPN=10
            ;;
        lj)
            PPN=5
            ;;
        *)
            PPN=16
            ;;
    esac
elif [[ $QUEUENAME == "debug" ]]; then
    echo "ERROR: debug queue only available in Conte Cluster. Exiting ..."
    exit 1
fi



START_TEST=1

SCRIPTSDIR=$D3TSDIR/src/mab/scripts/run
while [[ $START_TEST -le $NTESTS ]]; do

    # determine end test index
    if [[ $((START_TEST+PPN-1)) -gt $NTESTS ]]; then
        END_TEST=$NTESTS
        PPN=$((END_TEST-START_TEST+1))
    else
        END_TEST=$((START_TEST+PPN-1))
    fi
    MEM="$((PPN*MEMPERTSK))gb"

    TEST_RANGE="$START_TEST $END_TEST"
    TEST_RANGE_STR=`printf "%03d_%03d" $START_TEST $END_TEST`
    OUTFILE=${SCRIPTSDIR}/${MSC}.${SCOREMAPPER}.${SETTINGS}.${FEATURESET}.${DATASET}.${TEST_RANGE_STR}.sh

    cat $SCRIPTSDIR/script_template.sh | sed \
        -e "s/NODES/$NODES/" \
        -e "s/PPN/$PPN/" \
        -e "s/QUEUENAME/$QUEUENAME/" \
        -e "s/WALLTIME/$WALLTIME/" \
        -e "s/MEM/$MEM/" \
        -e "s/DATASET/$DATASET/" \
        -e "s/TEST_RANGE/$TEST_RANGE/" \
        -e "s/MSC/$MSC/" \
        -e "s/SCOREMAPPER/$SCOREMAPPER/" \
        -e "s/FEATURESET/$FEATURESET/" \
        -e "s/SETTINGS/$SETTINGS/" \
        -e "s+D3TSDIR+$D3TSDIR+" \
        > $OUTFILE

    echo qsub $OUTFILE
    qsub $OUTFILE

    START_TEST=$((END_TEST+1))
done

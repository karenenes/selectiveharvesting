#!/bin/bash

# simulation parameters
if [ "$#" -ne 0 ]; then
    echo "Illegal number of parameters"
    exit 1
fi


for MSC in rr dts.5; do
for SETTINGS in 00011 00101 00110 00111 01001 01010 01011 01100 01101 01110 01111 10001 10010 10011 10100 10101 10110 10111 11000 11001 11010 11011 11100 11101 11110; do
for DATASET in donors dbpedia wikipedia citeseer kickstarter; do


# cluster parameters
NODES=1
PPN=20
QUEUENAME=debug
QUEUENAME=ribeirob
QUEUENAME=standby
WALLTIME="4:00:00"
MEM=200gb

SCOREMAPPER="max"
FEATURESET="all"
NTESTS=40


# dataset specific settings
MAXTESTS=1000
case $DATASET in
    dbpedia)
        MAXTESTS=725
	WALLTIME="1:15:00"
        MEMPERTSK=3
	;;
    citeseer)
        MAXTESTS=1583
	WALLTIME="1:15:00"
        MEMPERTSK=3
	;;
    wikipedia)
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
    dblp)
        MAXTESTS=7556
        MEMPERTSK=5
	;;
    lj)
        MAXTESTS=1441
        MEMPERTSK=10
	;;
    kickstarter)
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
    OUTFILE=${MSC}.${SCOREMAPPER}.${SETTINGS}.${FEATURESET}.${DATASET}.${TEST_RANGE_STR}.sh

    cat script_template_config2.sh | sed \
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
        > $OUTFILE

    echo qsub `pwd`/$OUTFILE
    qsub $OUTFILE

    START_TEST=$((END_TEST+1))
done

done
done
done

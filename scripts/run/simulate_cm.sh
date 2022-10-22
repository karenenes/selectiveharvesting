# -- karen

#!/bin/bash

# Make sure env var D3TSDIR is properly set
if [ x$D3TSDIR == x ]; then
    echo "ERROR: environment variable D3TSDIR not defined."
    exit 1
elif [ ! -f $D3TSDIR/src/mab/main.R ]; then
    echo "ERROR: src/mab/main.R not found. $D3TSDIR is not a valid directory."
    exit 2
fi

# get number of threads
export NTHREADS=1
if [ "$#" -eq 1 ]; then
    NTHREADS=$1
    echo "Number of threads set to ${NTHREADS}."
fi

MSC="rr"
SCOREMAPPER="max"
#FEATURESET="all"
FEATURESET="only_attrib_features"
#FEATURESET="only_structural"
#NTESTS=10 
NTESTS=10

for DATASET in citeseer dbpedia wikipedia donors kickstarter; do
#for DATASET in blogcatalog cora dblp flickr lj; do
#for DATASET in citeseer dbpedia wikipedia donors kickstarter blogcatalog cora dblp flickr lj; do
#for DATASET in donors; do
#for SETTINGS in ewls glm svr svrc; do
for SETTINGS in ewls; do

echo "**************************************************"
echo "Simulating method:$SETTINGS on dataset:$DATASET..."
echo "**************************************************"

# dataset specific settings
MAXTESTS=10
case $DATASET in
    dbpedia)
        MAXTESTS=725
	;;
	  cora)
        MAXTESTS=1089
	;;
	  flickr)
        MAXTESTS=9802
	;;
    citeseer)
        MAXTESTS=1583
	;;
    wikipedia)
        MAXTESTS=202
	;;
    blogcatalog)
        MAXTESTS=1423
	;;
    donors)
        MAXTESTS=56
	;;
    dblp)
        MAXTESTS=7556
	;;
    lj)
        MAXTESTS=1441
	;;
    kickstarter)
        MAXTESTS=1457
	;;
    *)
        echo "Invalid dataset $1"
        quit
        ;;
esac

if [ $NTESTS -gt $MAXTESTS ]; then
    echo "Overriding NTESTS; now set to $MAXTESTS"
    NTESTS=$MAXTESTS
fi

# "acconte" for the fact that conte has only 16 cores 
HOSTNAME=`hostname | cut -d- -f1`


START_TEST=1
END_TEST=$NTESTS


#source /etc/profile.d/modules.sh
#module load r
#module load octave
#module load python
#echo " "
#echo " "
echo "Job started on `hostname` at `date`"


NSEEDS=1
COLDPULLS=20
CODEDIR=$D3TSDIR/src/mab/
OUTPATH=$CODEDIR/results/${MSC}_${SCOREMAPPER}_${SETTINGS}_${FEATURESET}/
PARALLEL_UPDATES=0
HEURISTIC=mod


if [ ! -d "$OUTPATH" ]; then
    echo "Creating output folder $OUTPATH"
    mkdir -p $OUTPATH
else
    echo "Saving results to $OUTPATH"
fi

cd $CODEDIR

Rscript scripts/run/test.R $DATASET $START_TEST $END_TEST $NSEEDS $COLDPULLS $HEURISTIC $FEATURESET $MSC $SCOREMAPPER inits/settings/${SETTINGS}.settings.init.R $PARALLEL_UPDATES $OUTPATH $NTHREADS

#
#    End user code
#
echo "Job ended on `hostname` at `date`"

cd -

done
done

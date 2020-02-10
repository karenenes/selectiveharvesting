#!/bin/bash
#
#PBS -l nodes=NODES:ppn=PPN
#
#PBS -q QUEUENAME
#PBS -l walltime=WALLTIME
#PBS -l naccesspolicy=shared
#PBS -l mem=MEM
#
#
#PBS -j oe
#

source /etc/profile.d/modules.sh
module load r/3.1.0
module load octave
module load python
module list

ls -l /apps/rhel6/r/3.1.0/bin/Rscript
which Rscript

echo " "
echo " "
echo "Job started on `hostname` at `date`"
#
#    User code starts now
#

echo "Replaced placeholders:"
echo NODES
echo PPN
echo QUEUENAME
echo WALLTIME
echo MEM
echo DATASET
echo TEST_RANGE
echo MSC
echo SCOREMAPPER
echo FEATURESET
echo SETTINGS

NSEEDS=1
COLDPULLS=20
OUTPATH=$RCAC_SCRATCH/p2r_results/MSC_SCOREMAPPER_SETTINGS_FEATURESET/
HEURISTIC=mod
PARALLEL_UPDATES=0

CODEDIR=D3TSDIR/src/mab/
cd $CODEDIR

if [[ ! -d "$OUTPATH" ]]; then
    echo "Creating output folder $OUTPATH"
    mkdir -p $OUTPATH
else
    echo "Saving results to $OUTPATH"
fi

Rscript scripts/run/test.R DATASET TEST_RANGE $NSEEDS $COLDPULLS $HEURISTIC FEATURESET MSC SCOREMAPPER inits/settings/SETTINGS.settings.init.R $PARALLEL_UPDATES $OUTPATH PPN

#
#    End user code
#
echo "Job ended on `hostname` at `date`"

#!/bin/bash
#   Request 1 processors on 1 node
#
#PBS -l nodes=NODES:ppn=PPN
#
#   Request 10 hours of walltime
#   Request 50 gigabytes of memory per process
#
#PBS -q QUEUENAME
#PBS -l walltime=WALLTIME
#PBS -l naccesspolicy=shared
#PBS -l naccesspolicy=singleuser
#PBS -l mem=MEM
#
#
#PBS -j oe
#
#   The following is the body of the script. By default,
#   PBS scripts execute in your home directory, not the
#   directory from which they were submitted. The following
#   line places you in the directory from which the job
#   was submitted.
#

source /etc/profile.d/modules.sh
#module load devel
#module load python
#module load anaconda
#module load hdf5/1.8.13_gcc-4.7.2
module load r
module load octave
module load python
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
OUTPATH=$RCAC_SCRATCH/p2r_results/config2_binary/MSC_SCOREMAPPER_SETTINGS_FEATURESET/
HEURISTIC=mod
PARALLEL_UPDATES=0

CODEDIR=~/pay2recruit/project/src/mab/
cd $CODEDIR

if [[ ! -d "$OUTPATH" ]]; then
    echo "Creating output folder $OUTPATH"
    mkdir -p $OUTPATH
else
    echo "Saving results to $OUTPATH"
fi

Rscript scripts/test.R DATASET TEST_RANGE $NSEEDS $COLDPULLS $HEURISTIC FEATURESET MSC SCOREMAPPER inits/settings/config2_binary/SETTINGS.settings.init.R $PARALLEL_UPDATES $OUTPATH PPN

#
#    End user code
#
echo "Job ended on `hostname` at `date`"

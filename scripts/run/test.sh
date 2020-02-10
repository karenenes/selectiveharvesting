#!/bin/bash

DATASET="$1"
TEST_FIRST="$2"
TEST_FINAL="$3"
NPROC="$4"

#  dataset            = raw_args[1],
#  test_range_start   = as.numeric(raw_args[2]),
#  test_range_end     = as.numeric(raw_args[3]),
#  nseeds             = as.numeric(raw_args[4]),
#  ncold              = as.numeric(raw_args[5]),
#  heuristic          = raw_args[6],
#  feature_set        = raw_args[7],
#  action_selector    = raw_args[8],
#  score_mapper       = raw_args[9],
#  settings_init      = raw_args[10],
#  parallel_updates   = as.logical(as.numeric(raw_args[11])),
#  output_path        = raw_args[12],
#  nproc              = as.numeric(raw_args[13]))


#Rscript scripts/test.R  $DATASET  $TEST_FIRST  $TEST_FINAL  1  20  mod all  ucb1 max  inits/settings/activesearch.settings.init.R 0   output/activesearch/   $NPROC
Rscript scripts/test.R  donors 1 1 1  20  mod all dts.5 max inits/settings/config1.settings.init.R 0   output/dts.5_max_config1/   1



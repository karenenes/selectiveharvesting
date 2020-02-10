###
printArgument <- function(name, value)
{
  cat(sprintf("%17s = %s\n", name, as.character(value)))
}

###
confirmArguments <- function(proc_args)
{
  cat("\n")
  for (arg in names(proc_args)) {
    printArgument(arg, proc_args[[arg]])
  }
#   cat("\nPress [enter] to run")
#   scan("stdin", raw(), n=1, blank.lines.skip=FALSE)
  cat("\n\n")
}

###############################################################################

raw_args <- commandArgs(trailingOnly = TRUE)

# main <- function(problem_name, test_range=NA, nseeds=NA, ncold=NA, heuristic=NA, feature_set=NA, action_selector=NA, score_mapper=NA,
#                  settings_init=NA, parallel_updates=NA, output_path=NA, nproc=1, output_to_file=FALSE, observation_file=NA)
proc_args <- list(
  dataset            = raw_args[1],
  test_range_start   = as.numeric(raw_args[2]),
  test_range_end     = as.numeric(raw_args[3]),
  nseeds             = as.numeric(raw_args[4]),
  ncold              = as.numeric(raw_args[5]),
  heuristic          = raw_args[6],
  feature_set        = raw_args[7],
  action_selector    = raw_args[8],
  score_mapper       = raw_args[9],
  settings_init      = raw_args[10],
  parallel_updates   = as.logical(as.numeric(raw_args[11])),
  output_path        = raw_args[12],
  nproc              = as.numeric(raw_args[13]))

cat(sprintf('source("main.R");main("%s",test_range=c(%d,%d),nseeds=%d,ncold=%d,heuristic="%s",feature_set="%s",action_selector="%s",score_mapper="%s",settings_init="%s",parallel_updates=%s,output_path="%s",nproc=%d)\n',
     proc_args$dataset,
     proc_args$test_range_start, proc_args$test_range_end,
     proc_args$nseeds,
     proc_args$ncold,
     proc_args$heuristic,
     proc_args$feature_set,
     proc_args$action_selector,
     proc_args$score_mapper,
     proc_args$settings_init,
     proc_args$parallel_updates,
     proc_args$output_path, 
     proc_args$nproc))

confirmArguments(proc_args)

source("main.R")
main(proc_args$dataset,
     test_range=c(proc_args$test_range_start, proc_args$test_range_end),
     nseeds=proc_args$nseeds,
     ncold=proc_args$ncold,
     heuristic=proc_args$heuristic,
     feature_set=proc_args$feature_set,
     action_selector=proc_args$action_selector,
     score_mapper=proc_args$score_mapper,
     settings_init=proc_args$settings_init,
     parallel_updates=proc_args$parallel_updates,
     output_path=proc_args$output_path, 
     nproc=proc_args$nproc)


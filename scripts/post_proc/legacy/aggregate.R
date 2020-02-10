##
 # Aggregates all results lists loaded from the files and writes final output files.
 ##
main <- function(filenames, output_path)
{
  type <- NA
  settings <- NULL
  seeds <- list()
  aggregated <- list()
  run_times <- NULL
  
  # Loads all results files and appends the results.
  cat("Loading results files:\n")
  file_index <- 1
  for (filename in filenames) {
    cat(sprintf("\tFile %d of %d\r", file_index, length(filenames)))
    load(filename)
    stopifnot(exists("results"))
    file_index <- 1 + file_index
    
    # Validates the settings.
    if (is.null(settings)) {settings <- results$settings}
    else {compareSettings(settings, results$settings)}
    
    # Gets the test type.
    test_type <- strsplit(tail(strsplit(filename, "/", fixed=TRUE)[[1]], n=1), "_")[[1]][2]
    if (is.na(type)) {type <- test_type}
    else {stopifnot(type == test_type)}
    
    # Appends the results and seeds.
    run_index <- 1
    initial_index <- 1 + length(aggregated)
    final_index <- initial_index + length(results) - 2
    for (aggregated_index in initial_index:final_index) {
      # Appends the results.
      aggregated[[aggregated_index]] <- results[[run_index]]
      # Appends the seeds.
      seeds[[aggregated_index]] <- results[[run_index]]$seeds
      # Increments total run time.
      initial_run_time <- results[[run_index]]$initial_time
      final_run_time   <- results[[run_index]]$final_time
      run_times <- c(run_times, as.numeric(difftime(final_run_time, initial_run_time, units="secs")))
      # Indexes the next run.
      run_index <- 1 + run_index
    }
    # Removes this file's object.
    rm(results)
  }
  
  # Asserts seeds do not overlap.
  cat("\nSeed counts:\n")
  seeds_table <- table(unlist(seeds))
  print(seeds_table)
  stopifnot(max(seeds_table) == 1)
  
  # Calculates mean run time.
  cat(sprintf("\nMean run time: %.3f ± %.3f (secs)\n", mean(run_times), sd(run_times)))
  
  # Writes output files using the aggregated results.
  cat("Processing aggregated tests...\n")
  saveResults(output_path, aggregated, settings, type)
}


##
 # Writes output files.
 ##
saveResults <- function(output_path, tests, settings, type)
{
  ntests <- length(tests)
  steps <- 1:settings$nattempts
  output_file <- sprintf("%s%s", output_path, getOutputFilename(settings, ntests, type))
  
  # Saves general simulation results.
  sink(file=sprintf("%s.txt", output_file))
  checkTests(tests, steps); cat("\n")
  sink()
  
  # Gets turn-by-turn data from simulation results.
  list_npositive <- list()
  list_recruited <- list()
  list_models    <- list()
  list_obs       <- list()
  for(i in 1:length(tests)) {
    list_models[[i]]    <- unlist(Map(function(x) x[['model_id']], tests[[i]]$turns))
    list_obs[[i]]       <- do.call(rbind, Map(function(x) x[['observed_feats']], tests[[i]]$turns))
    list_npositive[[i]] <- unlist(Map(function(x) x[['total_payoff']], tests[[i]]$turns))
    list_recruited[[i]] <- unlist(Map(function(x) x[['recruited_id']], tests[[i]]$turns))
    rownames(list_obs[[i]]) <- list_recruited[[i]]
  }
  list_models$names <- unlist(Map(function(i) settings$models[[i]]$getName(), 1:settings$nmodels))
  
  # Stores the lists on R data files.
  cat("Saving aggregated results data...\n")
  save(list_npositive, file=paste(output_file, '.npositive.RData', sep=""))
  save(list_recruited, file=paste(output_file, '.recruited.RData', sep=""))
  save(list_models,    file=paste(output_file, '.models.RData', sep=""))
  if(is.na(settings$observation_file))  # do not save obs if used observation file
    save(list_obs, file=paste(output_file, '.obs.RData', sep=""))
}


##
 # Returns a character string containing a name for the results file.
 ##
getOutputFilename <- function(settings, ntests, type)
{
  return (sprintf("%s_t%d_a%d_m%d_s%d_%s", 
                  settings$problem$name, ntests, settings$nattempts,
                  settings$nmodels, settings$nseeds, type))
}


##
 #
 ##
checkTests <- function(tests, steps)
{
  for (i in 1:length(steps)) {
    cat(paste("Results at turn ", steps[i], ":\n", sep=""))
    
    total_payoffs <- getResultVector(tests, steps[i], "total_payoff")
    mean_payoffs <- getResultVector(tests, steps[i], "mean_payoff")
    nrecruiteds <- getResultVector(tests, steps[i], "nrecruited")
    
    cat("\tTotal payoff:\n")
    printResults(total_payoffs)
    cat("\tMean payoff:\n")
    printResults(mean_payoffs)
    cat("\tTotal number of recruited nodes:\n")
    printResults(nrecruiteds)
    cat("\n")
  }
}

##
 #
 ##
printResults <- function(results_vector)
{
  stats <- round(c(mean(results_vector), sd(results_vector)), 4)
  cat(paste("\t\t", stats[1], " Â± ", stats[2], "\n", sep=""))
}

##
 #
 ##
getResultVector <- function(tests, step, result)
{
  return (unlist(Map(function(x) x$turns[[as.character(step)]][[result]], tests)))
}


##
 # Compares two Settings instances and throws an error if they seem to contain
 # different configurations.
 ##
compareSettings <- function(settings1, settings2)
{
  stopifnot(settings1$ncold == settings2$ncold)
  stopifnot(settings1$nseeds == settings2$nseeds)
  stopifnot(settings1$nmodels == settings2$nmodels)
  stopifnot(settings1$rng_seed == settings2$rng_seed)
  stopifnot(settings1$nattempts == settings2$nattempts)
  stopifnot(settings1$problem$name == settings2$problem$name)
  stopifnot(settings1$parallel_updates == settings2$parallel_updates)
  stopifnot(settings1$initial_heuristic == settings2$initial_heuristic)
  stopifnot(settings1$model_selection_method == settings2$model_selection_method)
  stopifnot((is.na(settings1$observation_file) & is.na(settings2$observation_file))
            | (settings1$observation_file == settings2$observation_file))
  for (i in 1:settings1$nmodels) {
    stopifnot(settings1$models[[i]]$getName() == settings2$models[[i]]$getName())
  }
}

###############################################################################

##
 # Allows this script to be run with Rscript.
 # First argument must be the output path for the final results file.
 # All following arguments must be RData results files to be aggregated.
 ##
raw_args <- commandArgs(trailingOnly = TRUE)
output_path <- raw_args[1]
filenames <- raw_args[2:length(raw_args)]
main(filenames, output_path)


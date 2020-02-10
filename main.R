source("inits/features.init.R")
source("inits/score_mappers.init.R")
source("inits/selector.init.R")
source("simulator.R")

library(doMC)

options(warn=-1)
options(error=recover)

# -- Karen
Nturn <<- 0  # num da iteracao atual -- snapshots
# -- 

##
  # Sample main() call:
  # main("donors", c(1,3), 1, 20, "mod", "all", "rr", "max", "inits/settings/test.settings.init.R", FALSE, "/tmp/", 1)
 ##

main <- function(problem_name, test_range=NA, nseeds=NA, ncold=NA, heuristic=NA, feature_set=NA, action_selector=NA, score_mapper=NA,
                 settings_init=NA, parallel_updates=NA, output_path=NA, nproc=1, output_to_file=FALSE, observation_file=NA, save_observations=FALSE)
{
  # Loads the settings file.
  if (is.na(settings_init)) {source("inits/settings/default.settings.init.R")}
  else {source(settings_init)}
  
  # Sanity checks.
  stopifnot(problem_name %in% names(SETTINGS))
  stopifnot(feature_set %in% names(FEATURE_SETS))
  stopifnot(action_selector %in% names(SELECTORS))
  stopifnot(score_mapper %in% names(SCORE_MAPPERS))
  stopifnot((!is.na(output_path)) & is.character(output_path))
  stopifnot(!is.na(output_to_file) && is.logical(output_to_file))
  stopifnot(is.na(observation_file) || (!is.na(observation_file) && !save_observations))
  stopifnot(system(sprintf('if [ -d %s ]; then exit 0; else exit 1; fi', output_path)) == 0)
  stopifnot(is.numeric(test_range) & (length(test_range) == 2) & (test_range[2] >= test_range[1]))
  
  # Registers the parallel backend.
  registerDoMC(nproc) 

  # Sets simulation settings.
  settings <- SETTINGS[[problem_name]]$clone()
  settings$action_selector <- SELECTORS[[action_selector]]$clone()
  settings$score_mapper <- SCORE_MAPPERS[[score_mapper]]$clone()
  settings$feature_set <- FEATURE_SETS[[feature_set]]$clone()
  settings$save_observations <- save_observations
  settings$observation_file <- observation_file
  settings$parallel_updates <- parallel_updates
  settings$output_to_file <- output_to_file
  settings$initial_heuristic <- heuristic
  settings$nseeds <- nseeds
  settings$ncold <- ncold

  # Prints the selected models, score mapper and action selector.
  cat("\nModels:\n")
  Map(function(i) cat(sprintf("\t%s\n", settings$models[[i]]$getName())), 1:settings$nmodels)
  cat(sprintf("\nAction selection policy: %s\n", settings$action_selector$getName()))
  cat(sprintf("Score mapper: %s\n\n", settings$score_mapper$getName()))
  
  # Loads the graph.
  graph <- loadFullGraph(settings$problem)

  # Checks whether simulation progress should be written to file instead of the screen.
  output_filename <- getOutputFilename(problem_name, settings_init, test_range, feature_set, action_selector, score_mapper)
  if (settings$output_to_file) {
    temp_file <- sprintf("%s%d_%s.temp.txt", output_path, Sys.getpid(), output_filename)
    cat(sprintf("\nProgress sunk to file %s.\n", temp_file))
    sink(file=temp_file)
  }
  
  ## --Karen
  # Gets the problem name as a global variable. 
  dataName <<- problem_name
  
  # Runs all tests.
  results <- runTests(graph, settings, test_range, output_path)
  
  # Resumes writing to screen.
  if (settings$output_to_file) {sink()}
  
  # Stores the results.
  cat("\nSimulation finished.\n")
  cat(sprintf("Valid tests: %d out of %d.\n", results$valid_tests, results$ntests_requested))
  cat("Saving tests to RData file...\n")
  save(results, file=sprintf("%s%s.RData", output_path, output_filename))
}


##
 #
 ##
runTests <- function(graph, settings, test_range, output_path)
{
  set.seed(settings$rng_seed)
  
  # Calculates the number of tests and total number of seeds needed.
  targets <<- getTargets(graph)
  targets <- sample(targets, length(targets))
  ntests <- 1 + test_range[2] - test_range[1]
  total_seeds <- ntests * settings$nseeds
  stopifnot(total_seeds <= length(targets))
  
  # Selects the set of seeds for the specified tests.
  initial_seed <- 1 + (test_range[1] - 1) * settings$nseeds
  final_seed <- initial_seed + total_seeds - 1
  seeds <- targets[initial_seed:final_seed]
  seeds <- split(seeds, ceiling(seq_along(seeds) / settings$nseeds))
  
  # If a file is specified, loads observations to list_obs.
  if(!is.na(settings$observation_file)) {
    cat(sprintf("Loading observation file %s...\n", settings$observation_file))
    load(settings$observation_file)
  }

  # Runs all tests.
   tests = list()  # Sequential.
   for (i in 1:ntests) { # Sequential.
    #tests <- foreach(i=1:ntests, .errorhandling="remove") %dopar% {  # Parallel.
     
    #feature_importance <<- c() # -- karen 
    #selected_models <<- list(c(), c()) # -- 
    #meiuca_interval <<- list(c(), c()) # --
    #meiuca_size <<- list(c(), c()) # --
    #recruited_node_model <<- list(c(), c()) # --
    #error_yhat <<- c() # --
    #error_yhat_2 <<- c() # -- karen
    
   
   # Loads exogenous training data if requested.
    test_idx <- test_range[1] + i - 1
    external_feats <- if (exists("list_obs")) list_obs[[test_idx]] else NULL

    # Runs the simulation.
    simulator <- Simulator$new(sprintf("Test %d", i), graph, settings, external_feats)
    initial_time <- strptime(Sys.time(), format="%Y-%m-%d  %H:%M:%S")
    simulator$simulate(seeds[[i]])
    final_time <- strptime(Sys.time(), format="%Y-%m-%d  %H:%M:%S")
    
    # Retrieves simulation results for each turn.
    cat(sprintf("\nTest %02d: retrieving simulation results...\n", i))
    steps <- 1:settings$nattempts
    turns <- Map(function(x) list(), steps)
    names(turns) <- steps
    for (j in 1:length(steps)) {
      turns[[j]] <- simulator$getSimulationResultsAtTurn(steps[j])}
    
    # Builds the test results list.
    results <- list(turns = turns,
                    initial_time = initial_time,
                    final_time = final_time,
                    #feature_importance = feature_importance, # -- karen
                    seeds = seeds[[i]])
              
                     
                    #selected_models = selected_models, # -- 
                    #meiuca_interval = meiuca_interval, # --
                    #meiuca_size = meiuca_size, # --
                    #recruited_node_model = recruited_node_model, # --
                    #error_yhat = error_yhat, # --
                    #error_yhat_2 = error_yhat_2) # -- karen
    
    #results  # Parallel.
    tests[[i]] <- results  # Sequential.
  }
  
  # Builds the results list with results from all runs and other data.
  retval <- list()
  retval$runs <- tests
  retval$test_range <- test_range
  retval$ntests_requested <- ntests
  retval$valid_tests <- length(tests)
  retval$settings <- settings$clone()
  
  return (retval)
}


##
 # Loads the full graph from file.  
 ##
loadFullGraph = function(problem) 
{
  graph <- new(Graph, undirected=TRUE)
  # Asserts a valid Graph implementation from Pay2Recruit is being used.
  stopifnot(graph$getVersion() == "1.21")
  # Initializes the underlying graph.
  graph$readGraph(problem$graph_file)
  graph$readCommunities3(problem$attrib_file, 
                         problem$accept_trait, 
                         problem$accept_trait, 
                         1.0, 0.0, 1.0, 1.0, 
                         problem$amount_file)

  return (graph)
}


##
 # Returns the IDs of all target nodes.
 ##
getTargets = function(graph)
{
  kk <- as.character(sort(graph$getTargetNodes(graph$getTargetSize())))
  #save list of targets -- snapshots
  #aa <- as.vector(as.numeric(kk))
  #write.table(aa, file = paste("targets/", dataName,"_targets.txt", sep = ""), row.names = F, col.names = F)
  #browser()
  
  return (kk)
}


##
 # Returns a character string containing a name for the results file.
 ##
getOutputFilename <- function(problem_name=NA, settings_init=NA, test_range=NA, feature_set=NA, action_selector=NA, score_mapper=NA)
{
  settings_name <- strsplit(tail(strsplit(settings_init, "/", fixed=TRUE)[[1]], n=1), ".", fixed=TRUE)[[1]][1]
  strings <- gsub("_", "-", c(problem_name, settings_name, feature_set, action_selector, score_mapper))
  fn <- sprintf("%s_%s_%s_%s_%s_%d-%d", strings[1], strings[2], strings[3], strings[4], strings[5], test_range[1], test_range[2])
  return (fn)
}


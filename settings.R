library(R6)

##
 # Simulation and test settings.
 ##
Settings <- R6Class("Settings",

    public = list(

        problem = NA,                 # (Problem) The initialized problem instance.
        rng_seed = NA,                # (integer) Seed for the RNG. System time if NULL.
        nattempts = NA,               # (integer > 0) Total number of recruitments.
        
        nmodels = NA,                 # (integer > 0) The number of models.
        nseeds = NA,                  # (integer > 0) The number of seeds.
        ncold = NA,                   # (integer >= 0) Number of cold-start pulls.
        
        models = NA,                  # (list) List of initialized models to be used.
        feature_set = NA,             # (Pay2RecruitFeatures) Initialized feature set instance.
        score_mapper = NA,            # Initialized instance of a supported score mapper.
        action_selector = NA,         # Initialized instance of a supported model selection policy.
        
        initial_heuristic = NA,       # (character) Border node selection method until models are ready.
        parallel_updates = NA,        # (logical) Whether to update all models which recommend the selected node.
        
        output_to_file = NA,          # (logical) Whether to write simulation progress to file instead of screen.
        observation_file = NA,        # (NA/character) File containing training data to be used in place of observations.
        save_observations = NA,       # (logical) Whether to save the observed design matrix to file.
        
        ##
         # Initializes a new Settings instance.
         ##
        initialize = function(problem, rng_seed, nattempts, models)
        {
          self$rng_seed <- if (!is.null(rng_seed)) rng_seed else as.numeric(Sys.time())
          self$problem <- problem
          self$nattempts <- nattempts
          self$models <- models
          self$nmodels <- length(models)
        },
        
        ##
         # Validates a fully initialized Settings instance.
         ##
        validate = function()
        {
          stopifnot(self$nattempts >= 0)

          stopifnot(self$nseeds > 0)
          stopifnot(self$nseeds <= self$problem$target_size)
          stopifnot(self$nmodels > 0)
          stopifnot((self$ncold >= 0) & (self$ncold <= self$nattempts))
          
          stopifnot(is.list(self$models))
          stopifnot(length(self$models) == self$nmodels)
          stopifnot(class(self$feature_set)[1] == "Pay2RecruitFeatures")
          
          stopifnot(self$initial_heuristic %in% c("random", "mod", "modunz"))
          stopifnot(!is.na(self$parallel_updates) && is.logical(self$parallel_updates))
          
          stopifnot(is.logical(self$output_to_file))
          stopifnot(is.na(self$observation_file) || is.character(self$observation_file))
          stopifnot(!is.na(self$save_observations) && is.logical(self$save_observations))

          return (TRUE)
        }
    )
)

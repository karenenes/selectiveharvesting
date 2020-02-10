library(R6)
library(nnet)

source("selectors/base/single_arm.R")

##
 # Epsilon-Greedy policy for MAB.
 ##
PolicyEG <- R6Class("PolicyEG",
                    inherit = SingleArmSelector,
                      
    public = list(
      
        ##
         # Creates a new selector instance.
         ##
        initialize = function(eps=0.1)
        {
          stopifnot((eps >= 0) && (eps <= 1))
          private$epsilon = eps
        },
      
        ##
         # Initializes the selector instance.
         ##
        init = function(settings=NA)
        {
          super$init(settings$score_mapper)
          private$infos <- Map(function(i) private$policy_data_factory$new(), 1:settings$nmodels)
        },
        
        ##
         # Returns the name of this policy.
         ##
        getName = function()
        {
          return (sprintf("epsilon-Greedy, eps=%s", private$epsilon))
        },
        
        ##
         # Selects a model from the list according to the epsilon-Greedy policy.
         ##
        selectModels = function(model_ids=NA, simulation_state=NA)
        {
          if (runif(1) < private$epsilon) {
            return (sample.int(length(model_ids), 1))
          } else {
            scores <- unlist(Map(function(x) private$getScore(x), private$infos[model_ids]))
            return (which.is.max(scores))
          }
        },
        
        ##
         # Registers the feedback for the given model.
         ##
        addFeedback = function(model_id=NA, correct=NA)
        {
          info <- private$infos[[model_id]]
          info$nused <- 1 + info$nused
          if (correct) info$ncorrect <- 1 + info$ncorrect
          else info$nwrong <- 1 + info$nwrong
        }
    ),
    
    private = list(
      
        epsilon = NA,   # (double) Exploration probability.
        infos = NA,     # (list<PolicyData>) Model-level statistics.
      
        ##
         # Returns an epsilon-Greedy score given the model statistics.
         ##
        getScore = function(info=NA)
        {
          score <- info$ncorrect / info$nused
          return (score)
        },
        
        # Instances hold model-level data fields for keeping track of policy statistics.
        policy_data_factory = R6Class("PolicyData", public = list(
            ncorrect = 1,
            nwrong = 1,
            nused = 2
        ))
    )
)


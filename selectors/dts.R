library(R6)
source("selectors/base/single_arm.R")

##
 # Thompson Sampling policy for MAB.
 ##
PolicyDTS <- R6Class("PolicyDTS",
                    inherit = SingleArmSelector,
                      
    public = list(
        initialize = function(C=1) {
          private$C = C
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
          return (sprintf("Dynamic Thompson Sampling, C = %.2f", private$C))
        },
        
        ##
         # Selects a model from the list according to the Thompson Sampling policy.
         ##
        selectModels = function(model_ids=NA, simulation_state=NA)
        {
          scores <- unlist(Map(function(x) private$getScore(x), private$infos[model_ids]))
          return (which.is.max(scores))
        },
        
        ##
         # Registers the feedback for the given model.
         ##
        addFeedback = function(model_id=NA, correct=NA)
        {
          info <- private$infos[[model_id]]
          info$nused <- 1 + info$nused
          reward = 1*correct
          C = private$C
          if( info$alpha + info$beta < private$C ) {
            info$alpha <- info$alpha + reward
            info$beta  <- info$beta + (1-reward)
          } else {
            info$alpha <- (info$alpha + reward)*(C/(C+1.0))
            info$beta  <- (info$beta + (1-reward))*(C/(C+1.0))
          }
        }
    ),
    
    private = list(
      
        infos = NA,  # (list<PolicyData>) Model-level statistics.
        C = NA,
      
        ##
         # Returns a Thompson Sampling score given the model statistics.
         ##
        getScore = function(info=NA)
        {
          score <- rbeta(1, info$alpha, info$beta)
          return (score)
        },
        
        # Instances hold model-level data fields for keeping track of policy statistics.
        policy_data_factory = R6Class("PolicyData", public = list(
                nused = 2,
                alpha = 1,
                beta = 1
            )
        )
    )
)


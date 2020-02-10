library(R6)
source("selectors/base/single_arm.R")

##
 # Testing Bruno's policy for MAB.
 ##
PolicyBruno <- R6Class("PolicyBruno",
                       inherit = SingleArmSelector,
                      
    public = list(
      
        ##
         # Creates a new selector instance.
         ##
        initialize = function(epsilon=0.1)
        {
          private$epsilon = epsilon
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
          return (sprintf("Bruno, eps=%.2f", private$epsilon))
        },
        
        ##
         # Selects a model from the list according to the Thompson Sampling policy.
         ##
        selectModels = function(model_ids=NA, simulation_state=NA)
        {
          graph <- simulation_state$observed_graph
          neighbor_counts <- graph$getNeighborsCount(graph$recruited)
          private$C <- max(2, max(c(1, var(neighbor_counts)), na.rm=TRUE) / max(1, mean(neighbor_counts)))
          
          if(runif(1) < private$epsilon) {
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
        epsilon = NA,
      
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


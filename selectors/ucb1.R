library(R6)
source("selectors/base/single_arm.R")

##
 # UCB1 policy for MAB.
 ##
PolicyUCB1 <- R6Class("PolicyUCB1",
                      inherit = SingleArmSelector,
                      
    public = list(
        
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
          return ("UCB1")
        },
        
        ##
         # Selects a model from the list according to the UCB1 policy.
         ##
        selectModels = function(model_ids=NA, simulation_state=NA)
        {
          scores <- unlist(Map(function(x) private$getScore(simulation_state$current_turn, x), private$infos[model_ids]))
          return (which.is.max(scores))
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
      
        infos = NA,  # (list<PolicyData>) Model-level statistics.
      
        ##
         # Returns the UCB1 score given the current turn and model statistics.
         ##
        getScore = function(turn=NA, info=NA)
        {
          upper_bound <- sqrt(2 * log(turn) / info$nused)
          score <- (info$ncorrect / info$nused) + upper_bound
          return (score)
        },
        
        # Instances hold model-level data fields for keeping track of policy statistics.
        policy_data_factory = R6Class("PolicyData", public = list(
                ncorrect = 1,
                nwrong = 1,
                nused = 2
            )
        )
    )
)


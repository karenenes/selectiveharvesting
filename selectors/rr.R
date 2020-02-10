library(R6)
source("selectors/base/single_arm.R")

##
 # Round-Robin policy for MAB.
 ##
PolicyRR <- R6Class("PolicyRR",
                    inherit = SingleArmSelector,
                    
    public = list(
      
        ##
         # Initializes the selector instance.
         ##
        init = function(settings=NA)
        {
          super$init(settings$score_mapper)
        },
        
        ##
         # Returns the name of this policy.
         ##
        getName = function()
        {
          return ("Round-Robin")
        },
        
        ##
         # Selects a model from the list according to the Round-Robin policy.
         ##
        selectModels = function(model_ids=NA, simulation_state=NA)
        {
          return (1 + (simulation_state$current_turn %% length(model_ids)))
        },
        
        ##
         # Registers the feedback for the given model.
         ##
        addFeedback = function(model_id=NA, correct=NA)
        {}
    )
)


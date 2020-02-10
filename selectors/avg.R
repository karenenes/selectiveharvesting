library(R6)

##
 # Round-Robin policy for MAB.
 ##
PolicyAVG <- R6Class("PolicyAVG",
                    
    public = list(
      
        ##
         # Initializes the selector instance.
         ##
        init = function(settings=NA)
        {
          private$score_mapper <- settings$score_mapper
        },
        
        ##
         # Returns the name of this policy.
         ##
        getName = function()
        {
          return ("Average")
        },
        
        ##
         # Selects a model from the list according to the Round-Robin policy.
         ##
        selectModels = function(model_ids=NA, simulation_state=NA)
        {
          private$model_ids <- model_ids
          return (1:length(model_ids))
        },
        
        
        ##
         # Selects an action according to the Average policy.
         ##
        selectAction = function(score_vectors=NA, simulation_state=NA)
        {
          stopifnot(length(score_vectors) == length(private$model_ids))

          score_sum <- rep(0,length(score_vectors[[1]]))
          for (i in 1:length(score_vectors))
            score_sum <- score_sum + private$score_mapper$buildDistribution(score_vectors[[i]], simulation_state)

          best <- which(score_sum == max(score_sum))
          action <- sample(x=names(best),size=1)
          
          return (list(vector_id = 1,
                       action_id = action))
        },
        
        ##
         # Registers the feedback for the given model.
         ##
        addFeedback = function(model_id=NA, correct=NA)
        {}
    ),

    private = list (
        score_mapper = NA,
        model_ids = NA
    )
)


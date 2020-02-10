library(R6)

##
 # Base class for policies which:
 #    1) select a single arm, then;
 #    2) draw an action by mapping the selected arm's scores to a probability distribution.
 ##
SingleArmSelector <- R6Class("SingleArmSelector",
                         
    public = list(
      
        ##
         # 
         ##
        init = function(score_mapper=NA)
        {
          private$score_mapper <- score_mapper
        },
        
        ##
         # Selects an action given a vector of action scores.
         ##
        selectAction = function(score_vectors=NA, simulation_state=NA)
        {
          stopifnot(length(score_vectors) == 1)
          probs <- private$score_mapper$buildDistribution(score_vectors[[1]], simulation_state)
          return (list(vector_id = 1,
                       action_id = names(probs)[sample.int(length(probs), 1, prob=probs)]))
        },
        
        ##
         #
         ##
        getAgreers = function(action_id=NA, score_vectors=NA, simulation_state=NA)
        {
          stopifnot(length(score_vectors) > 0)
          stopifnot(class(private$score_mapper)[1] == "ScoreMapperMaxScore")
          agreesWithAction <- function(v) {(private$score_mapper$buildDistribution(v, simulation_state)[action_id] > 0)}
          return (which(unlist(Map(function(v) all(is.numeric((v))) && agreesWithAction(v), score_vectors))))
        }
    ),
    
    private = list(
      
        score_mapper = NA
    )
)


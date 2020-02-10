library(R6)

##
 # 
 ##
ScoreMapperMaxScore <- R6Class("ScoreMapperMaxScore",
                         
    public = list(
        
        ##
         #
         ##
        buildDistribution = function(score_vector=NA, simulation_state=NA)
        {
          best <- which(score_vector == max(score_vector))
          dist <- rep(0, length(score_vector))
          dist[best] <- 1 / length(best)
          names(dist) <- names(score_vector)
          return (dist)
        },
        
        ##
         #
         ##
        getName = function()
        {
          return ("Maximum Score")
        }
    )
)


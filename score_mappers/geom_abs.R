library(R6)

##
 # 
 ##
ScoreMapperGeometricAbsolute <- R6Class("ScoreMapperGeometricAbsolute",
    public = list(
        
        ##
         # geom-abs (top 10)
         ##
        buildDistribution = function(score_vector=NA, simulation_state=NA)
        {
          p <- 1-exp(log(1-.70)/10)
          scores <- (1-p)^rank(-score_vector)*p
          dist <- scores/sum(scores)
          return (dist)
        },
        
        ##
         #
         ##
        getName = function()
        {
          return ("Geometric Absolute")
        }
    )
)


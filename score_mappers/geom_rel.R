library(R6)

##
 # 
 ##
ScoreMapperGeometricRelative <- R6Class("ScoreMapperGeometricRelative",
    public = list(
        
      ##
       # geom-rel (top 10%)
       ##
      buildDistribution = function(score_vector=NA, simulation_state=NA)
      {
        K <- length(score_vector)
        p <- 1-exp(log(1-.9)/(0.1*K))
        scores <- (1-p)^rank(-score_vector)*p
        dist <- scores/sum(scores)
        return (dist)
      },
      
      ##
       #
       ##
      getName = function()
      {
        return ("Geometric Relative")
      }
    )
)


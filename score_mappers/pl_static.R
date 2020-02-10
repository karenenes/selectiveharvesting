library(R6)

##
 # 
 ##
ScoreMapperPowerLawStatic <- R6Class("ScoreMapperPowerLawStatic",
    public = list(
        
      ##
       # pl-static
       ##
      buildDistribution = function(score_vector=NA, simulation_state=NA)
      {
        r <- 3.0
        scores <- rank(-score_vector)^(-r)
        dist <- scores/sum(scores)
        return (dist)
      },
      
      ##
       #
       ##
      getName = function()
      {
        return ("Power Law Static")
      }
    )
)


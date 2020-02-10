library(R6)

##
 # 
 ##
ScoreMapperGeometricDynamic <- R6Class("ScoreMapperGeometricDynamic",
    public = list(
        
      ##
       # geom-dyn (top 10 ... top 1)
       ##
      buildDistribution = function(score_vector=NA, simulation_state=NA)
      {
        curr_turn <- simulation_state$current_turn
        final_turn <- simulation_state$final_turn
        p <- 1-exp(log(1-.99)/max(1, 10*(1-curr_turn/final_turn)))
        scores <- (1-p)^rank(-score_vector)*p
        dist <- scores/sum(scores)
        return (dist)
      },
      
      ##
       #
       ##
      getName = function()
      {
        return ("Geometric Dynamic")
      }
    )
)


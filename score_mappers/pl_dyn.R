library(R6)

##
 # 
 ##
ScoreMapperPowerLawDynamic <- R6Class("ScoreMapperPowerLawDynamic",
    public = list(
        
      ##
       # pl-dyn (top 10 ... top 1)
       ##
      buildDistribution = function(score_vector=NA, simulation_state=NA)
      {
        K <- length(score_vector)
        curr_turn <- simulation_state$current_turn
        final_turn <- simulation_state$final_turn
        k <- max(1,as.integer(round(10*(1-curr_turn/final_turn))))
        r <- uniroot(function(x,K) {z <- (1:K)^(-x); return(.90-sum((z/sum(z))[1:k]))}, K=K, interval=c(1.0,10.0) )$root
        scores <- rank(-score_vector)^(-r)
        dist <- scores/sum(scores)
        return (dist)
      },
      
      ##
       #
       ##
      getName = function()
      {
        return ("Power Law Dynamic")
      }
    )
)


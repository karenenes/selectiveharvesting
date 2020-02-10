library(R6)

##
 # 
 ##
ScoreMapperPowerLawAbsolute <- R6Class("ScoreMapperPowerLawAbsolute",
    public = list(
        
      ##
       # pl-abs (top 10)
       ##
      buildDistribution = function(score_vector=NA, simulation_state=NA)
      {
        K <- length(score_vector)
        r <- uniroot(function(x,K) {z <- (1:K)^(-x); return(.9-sum((z/sum(z))[1:10]))}, K=K, interval=c(1.0,10.0) )$root
        scores <- rank(-score_vector)^(-r)
        dist <- scores/sum(scores)
        return (dist)
      },
      
      ##
       #
       ##
      getName = function()
      {
        return ("Power Law Absolute")
      }
    )
)


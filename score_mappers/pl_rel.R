library(R6)

##
 # 
 ##
ScoreMapperPowerLawRelative <- R6Class("ScoreMapperPowerLawRelative",
    public = list(
        
      ##
       # pl-rel (top 10%)
       ##
      buildDistribution = function(score_vector=NA, simulation_state=NA)
      {
        K <- length(score_vector)
        k <- as.integer(round(0.1*K))
        r <- uniroot(function(x,K) {z <- (1:K)^(-x); return(.9-sum((z/sum(z))[1:k]))}, K=K, interval=c(1.0,10.0) )$root
        scores <- rank(-score_vector)^(-r)
        dist <- scores/sum(scores)
        return (dist)
      },
      
      ##
       #
       ##
      getName = function()
      {
        return ("Power Law Relative")
      }
    )
)


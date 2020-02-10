library(R6)
library(RcppOctave)
o_addpath('models/sigmaopt/')


##
 #
 ##
SigmaOpt <- R6Class("SigmaOpt",

    public = list(
    
        ##
         # Initializes the model instance.
         # Default parameter values are those from LiblineaR.
         ##
        initialize = function(mu0 = .05, omega0 = .01, alpha0 = .1, obs_sigma = .25)
        {
          private$mu0 = mu0
          private$omega0 = omega0
          private$alpha = alpha0
          private$obs_sigma = obs_sigma
        },
        
        ##
         #
         ##
        isValid = function()
        {
          return (TRUE)
        },
        
        ##
        #
        ##
        getName = function()
        {
          name <- sprintf("SigmaOpt, mu=%.2f, omega=%.2f, alpha=.%2f, obs_sigma=%.2f",
                          private$mu0, private$omega0,
                          private$alpha0, private$obs_sigma)
          return (name)
        },
        
        ##
         #
         ##
        fit = function(x, y)
        {
        },
        
        ##
         # 
         ##
        predict = function(edges, isRecruited, responses)
        {
          ret <- .O$coreLoop(edges, private$mu0, private$omega0,
                             which(isRecruited), matrix(responses,ncol=1),
                             private$alpha, private$obs_sigma)
          yhat <- ret$score
          names(yhat) <- names(isRecruited)
          yhat <- yhat[!isRecruited]
          return(yhat)
        }
    ),

    private = list(
        model = NA,
        mu0 = NA,
        omega0 = NA,
        alpha = NA,
        obs_sigma = NA
    )
)

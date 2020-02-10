library(R6)
library(RcppOctave)
#source('node2vec/node-locality.R')

o_addpath('models/activesearch/')


##
 #
 ##
ActiveSearch <- R6Class("ActiveSearch",

    public = list(
    
        ##
         # Initializes the model instance.
         # Default parameter values are those from LiblineaR.
         ##
        initialize = function()
        {
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
          name <- sprintf("ActiveSearch")
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
          yhat <- .O$msgPassing(edges, isRecruited, responses, 200)
          names(yhat) <- names(isRecruited)
          yhat <- yhat[!isRecruited]
          return(yhat)
        }
    ),

    private = list(
        model = NA
    )
)

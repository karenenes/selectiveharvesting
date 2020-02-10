library(R6)

source('models/ewrls/internal.R')

##
 #
 ##
EWRLS <- R6Class("EWRLS",

    public = list(
    
        ##
         # Initializes the model instance.
         ##
        initialize = function(beta, lambda)
        {
          private$beta = beta
          private$lambda = lambda
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
          name <- sprintf("EWRLS, beta=%.2f, lambda=%.2f", private$beta, private$lambda)
          return (name)
        },
        
        ##
         #
         ##
        fit = function(x, y)
        {
          browser()
          private$model = InternalEWRLS$new(x, y, beta=private$beta, lambda=private$lambda)
        },
        
        ##
         #
         ##
        update = function(x, y)
        {
          private$model$update(x, y)
        },
        
        ##
         #
         ##
        predict = function(x)
        {
          return (private$model$predict(x)[,1])
        }
    ),

    private = list(

        model = NA,
        
        beta = NA,
        lambda = NA
    )
)

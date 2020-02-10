library(R6)

library(LiblineaR)

##
 #
 ##
Logistic <- R6Class("Logistic",

    public = list(
    
        ##
         # Initializes the model instance.
         # Default parameter values are those from LiblineaR.
         ##
        initialize = function(C=1, bias=TRUE)
        {
          private$heuristicC <- (C == "heuristic")
          private$C <- C
          private$bias <- bias
        },
        
        ##
         #
         ##
        isValid = function()
        {
          return (private$valid)
        },
        
        ##
        #
        ##
        getName = function()
        {
          name <- sprintf("Logistic Regression, C=%s, bias=%s", 
                          as.character(private$C),
                          as.character(private$bias))
          return (name)
        },
        
        ##
         #
         ##
        fit = function(x, y)
        {
          y = as.factor(y)
          if (length(levels(y)) > 1) {
            cost <- if (private$heuristicC) heuristicC(x) else private$C
            private$model <- LiblineaR(data=x, target=y, type=7, bias=private$bias, cost=cost)
            private$valid <- TRUE
          }
        },
        
        ##
         # 
         ##
        predict = function(x)
        {
          yhat <- predict(private$model, x, proba=TRUE)$probabilities[,"1"]
          names(yhat) <- rownames(x)
          return (yhat)
        }
    ),

    private = list(
        
        model = NA,
        
        heuristicC = NA,
        C = NA,
        bias = NA,
        valid = FALSE
    )
)

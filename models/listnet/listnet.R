library(R6)
library(RcppOctave)

#source('node2vec/node-locality.R')
o_addpath('models/listnet/')

##
 #
 ##
ListNet <- R6Class("ListNet",

    public = list(
    
        ##
         # Initializes an instance being instantiated.
         # Default parameter values are those from ListNet.
         ##
        initialize = function(max.iter=50, tol=5e-5)
        {
          private$max.iter <- max.iter
          private$tol <- tol
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
          name <- sprintf("ListNet, max.iter=%d, tol=%g", private$max.iter, private$tol)
          return (name)
        },

        ##
         #
         ##
        fit = function(x, y)
        {
          N <- length(y)
          if (N >= 5) {
            
            if (self$isValid()) {
              private$model <- .O$trainNN(matrix(1,N,1), x, as.matrix(y), private$max.iter, private$tol, TRUE, private$model)
            } else {
              private$model <- .O$trainNN(matrix(1,N,1), x, as.matrix(y), private$max.iter, private$tol, TRUE)
              private$valid <- TRUE
            }
          }
        },
        
        ##
         # 
         ##
        predict = function(x)
        {
          N <- nrow(x)
          
          if (self$isValid()) {
            yhat <- x %*% private$model
          } else {
            yhat <- rep(0,N)
          }
          names(yhat) <- rownames(x)
          return(yhat)
        }
    ),

    private = list(
        
        model = NA,
        emb_matrix = NA, 
        max.iter = NA,
        tol = NA,
        valid = FALSE
    )
)

library(R6)

library(randomForest)

##
 #
 ##
RForest <- R6Class("RForest",

    public = list(
    
        ##
         # Initializes the model instance.
         ##
        initialize = function(ntree=1000,mtry=NA)
        {
          private$ntree <- ntree
          private$mtry <- mtry
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
          name <- sprintf("RForest, ntree=%d", private$ntree)
          return (name)
        },
        
        ##
         #
         ##
        fit = function(x, y)
        {
          if (is.na(private$mtry)) {
            private$mtry = ceiling(sqrt(ncol(x)))
          }

          y = as.factor(y)
          if (length(levels(y)) > 1) {
              private$model = randomForest( x, y, ntree=private$ntree, mtry=private$mtry)
              private$valid = TRUE
          }

        },
        
        ##
         # 
         ##
        predict = function(x)
        {
          if (private$valid) {
              yhat <- predict( private$model, x, type="prob" )[,"1"]
          } else {
              yhat <- rep(0,nrow(x))
              names(yhat) <- rownames(x)
          }
          return(yhat)
        }
    ),

    private = list(
        
        model = NA,
        mtry = NA,
        ntree = NA,
        valid = FALSE
    )
)

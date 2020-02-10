library(R6)

library(adabag)

##
 #
 ##
AdaBoost <- R6Class("AdaBoost",

    public = list(
    
        ##
         # Initializes the model instance.
         # Default parameter values are those from LiblineaR.
         ##
        initialize = function(mfinal=100, coeflearn='Zhu', pickOne=FALSE, maxdepth=5)
        {
          private$mfinal <- mfinal
          private$coeflearn <- coeflearn
          private$pickOne <- pickOne
          private$maxdepth <- maxdepth
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
          name <- sprintf("AdaBoost, mfinal=%s, coeflearn=%s, pickOne=%s", 
                          as.character(private$mfinal),
                          as.character(private$coeflearn),
                          as.character(private$pickOne))
          return (name)
        },
        
        ##
         #
         ##
        fit = function(x, y)
        {
            if(length(y) > 10) {
            y = factor(y)
              result = tryCatch( {
                # must set minsplit <= 3, otherwise too slow to be used
                private$model <- boosting(formula = y ~ ., data = data.frame(x, y), boos=TRUE,
                coeflearn = private$coeflearn,
                mfinal = min(round(length(y)/3),private$mfinal),
                control = rpart.control(maxdepth=private$maxdepth,minsplit=3,maxcompete=0,maxsurrogate=0));
                private$valid <- TRUE
             },warning = function(w) print(w), error = function(e) print(e))
            }
        },
        
        ##
         # 
         ##
        predict = function(x)
        {
          if (!private$pickOne) {
            yhat <- predict.boosting(private$model, newdata=x)$prob[,2]
          } else {
            ensemble <- private$model$trees
            prob <- private$model$weights
            prob <- prob/sum(prob)
            tree.ind <- sample(length(ensemble), size=1, prob=prob)
            yhat <- predict(ensemble[[tree.ind]], newdata=data.frame(x), type="prob")[,2]
          }
          names(yhat) <- rownames(x)
          return (yhat)
        }
    ),

    private = list(
        
        model = NA,
        mfinal = NA,
        coeflearn = NA,
        valid = FALSE,
        pickOne = FALSE,
	maxdepth = NA
    )
)

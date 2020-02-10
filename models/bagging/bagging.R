library(R6)

library(adabag)

##
 #
 ##
Bagging <- R6Class("Bagging",

    public = list(
    
        ##
         # Initializes the model instance.
         ##
        initialize = function(mfinal=100, maxdepth=5, minsplit=3)
        {
          private$mfinal <- mfinal
          private$maxdepth <- maxdepth
          private$minsplit <- minsplit
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
          name <- sprintf("Bagging, mfinal=%s, maxdepth=%s, minsplit=%s", 
                          as.character(private$mfinal),
                          as.character(private$maxdepth),
                          as.character(private$minsplit))
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
                private$model <- bagging(formula = y ~ ., data = data.frame(x, y),
                mfinal = min(round(length(y)/3),private$mfinal),
                control = rpart.control(maxdepth=private$maxdepth,minsplit=private$minsplit,maxcompete=0,maxsurrogate=0));
                private$valid <- TRUE
             },warning = function(w) print(w), error = function(e) print(e))
            }
        },
        
        ##
         # 
         ##
        predict = function(x)
        {
            yhat <- predict.bagging(private$model, newdata=data.frame(x))$prob[,2]
          names(yhat) <- rownames(x)

          # save variables to file
          #ids = rownames(x)
          #pred_list = lapply(private$model$trees, function(model, x) {
          #    z <- predict(model, newdata=x, type="prob")[,2]
          #    names(z) <- c()
          #    return (z)
          #    }, data.frame(x) )
          #save(ids, pred_list, file=sprintf('yhat_%03d.RData', private$count))
          #private$count <- private$count+1

          return (yhat)
        }
    ),

    private = list(
        
        model = NA,
        count = 0,
        mfinal = NA,
        maxdepth = NA,
        minsplit = NA,
        valid = FALSE
    )
)

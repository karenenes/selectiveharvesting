library(R6)
library(rPython)
python.load("models/pyboosting/adaboost.py")
#library(RcppOctave)
#o_addpath('models/activesearch/')


##
 #
 ##
PyBoosting <- R6Class("PyBoosting",

    public = list(
    
        ##
         # Initializes the model instance.
         # Default parameter values are those from LiblineaR.
         ##
        initialize = function(nWL=10)
        {
          private$nWL = nWL
          python.exec(sprintf('model = AdaBoost(nWL=%d)',nWL))
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
          name <- sprintf("PyBoosting")
          return (name)
        },
        
        ##
         #
         ##
        fit = function(x, y)
        {
          if (length(y) >= 10) {
            python.method.call('model','fit', as.vector(x), y)
            private$valid <- TRUE
          }
          #if(length(y) == 149) {
          #  python.assign("X",as.vector(x))
          #  python.assign("y",y)
          #  python.exec("np.save('train_X',X)")
          #  python.exec("np.save('train_y',y)")
          #}
	  #private$counts = length(y)
          #print(sum(y))
        },

       browse = function()
       {
          browser()
       },
        
        ##
         # 
         ##
        predict = function(x)
        {
          yhat = python.method.call('model','predict', as.vector(x))
          #python.assign("X",as.vector(x))
          #python.exec(sprintf("np.save('test_X%03d',X)",private$counts))
          return(yhat)
        }

    ),

    private = list(
        #counts = 0,
        model = NA,
        nWL=NA,
        valid = FALSE
    )
)

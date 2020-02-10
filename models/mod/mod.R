library(R6)
##
 #
 ##
MOD <- R6Class("MOD",

    public = list(
    
        ##
         # Initializes the model instance.
         ##
        initialize = function()
        {},
        
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
          name <- sprintf("MOD")
          return (name)
        },
        
        ##
         #
         ##
        fit = function()
        {
          stop()
        },
        
        ##
         # 
         ##
        predict = function(x)
        {
          stop()
        }
    ),

    private = list()
)

library(R6)

##
 #
 ##
Random <- R6Class("Random",

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
          name <- sprintf("Random")
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

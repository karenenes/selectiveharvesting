# -- karen

library(R6)
library(glmnet)
library(adagio)
library(BBmisc)

source('models/ewls/internal.R')
#source('node2vec/node-locality.R')
##
 #
 ##

EWLS <- R6Class("EWLS",

    public = list(
    
        ##
         # Initializes the model instance.
         # Default parameter values are those from glmnet. 
         ##
        #},

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
          return (private$valid)
        },
        
        ##
        #
        ##
        
        getName = function() 
        {
          name <- sprintf("EWLS, beta=%.2f, lambda=%.2f", private$beta, private$lambda)
          return (name)

        },
    
     
        ##
         #
         ##
        fit = function(x, y){  
          
          y.factor = as.factor(y)
          if (length(levels(y.factor)) > 1 && min(table(y.factor))>2) {
              #result = tryCatch( {
              
              Nturn <<- Nturn + 1 # snapshots -- Karen  
              
              #node2vec
              #nodes <- as.vector(as.numeric(colnames(x)))
              #private$emb_matrix <- NA
              #private$emb_matrix <- get_node_locality(nodes_to_charge = nodes, problemName = dataName)
              #idx_app = match(nodes, private$emb_matrix[,1])
              #x <- rbind(x, t(private$emb_matrix[idx_app, -c(1)]))
              #--#
              private$model <- InternalEWLS$new(as.matrix(x), y, beta=private$beta, lambda=private$lambda)
              private$valid <- TRUE 
            
           #}, warning = function(w) print(w), error = function(e) print(e)) #trycatch
        } #if
      }, #fit
        
      ##
      # 
      ##
      ###
      
      update = function(x, y)
      {
        private$model$update(x, y)
      },
      
      ##
      # 
      ##
      ### 
      predict = function(x)
      {
        
        #node2vec
        #border_nodes <- as.vector(as.numeric(colnames(x)))
        #idx_app_border = match(border_nodes, private$emb_matrix[,1])
        #x <- rbind(x, t(private$emb_matrix[idx_app_border, -c(1)]))
        #--#
        
        yhat <- private$model$predict(x)[,1]
        
        names(yhat) <- colnames(x) 
        return (yhat) 
      }
  ),


    private = list(
        
        model = NA,
        model_2 = NA,
        gray_interval = NA,
        emb_matrix = NA, 

        prob = NA,
       
        heuristicC = NA,
        C = NA,
        bias = NA,
        beta = NA,
        lambda = NA,
        valid = FALSE
        
    )
)


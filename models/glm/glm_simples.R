# -- karen

library(R6)
library(glmnet)
library(adagio)
library(BBmisc)

source('node2vec/node-locality.R')
##
 #
 ##
GLM <- R6Class("GLM",

    public = list(
    
        ##
         # Initializes the model instance.
         # Default parameter values are those from glmnet.
         ##
        #},

        initialize = function() {},
        #initialize = function(alpha = 0)
        #{
        #  %private$alpha <- alpha
        #},
        
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
          name <- sprintf("Generalized linear model")
          return(name)

        },
       ##
         #
         ##
        fit = function(x, y){  
        
          y.factor = as.factor(y)
          if (length(levels(y.factor)) > 1 && min(table(y.factor))>2) {
              #result = tryCatch( {
            
              #node2vec
              #nodes <- as.vector(as.numeric(rownames(x)))
              #private$emb_matrix <- NA
              #private$emb_matrix <- get_node_locality(nodes_to_charge = nodes, problemName = dataName)
              #idx_app = match(nodes, private$emb_matrix[,1])
              #x <- cbind(x, private$emb_matrix[idx_app, -c(1)])
              #--#
              
              y0 = sample(which(y == 0))
              y1 = sample(which(y == 1))
              
              idx_training = c(y0[1:round(length(y0))], y1[1:round(length(y1))]) 
              
              #training set
              y_t = y[idx_training]
              x_t = as.matrix(x[idx_training, ])
             
              weights = c(rep(1,length(y_t)))
              
                
              #stratified foldid_1
              foldid_1 = c(sample(rep_len(1:3, length.out = length(which(y_t == 0)))), sample(rep_len(1:3, length.out = (length(which(y_t == 1))))))
              lambdas <- 10^seq(2, -5, by = -.1)

              private$model <- cv.glmnet(x = x_t, y = y_t, weights = weights, foldid = foldid_1, alpha = 0, lambda = lambdas, family = "binomial")
              
              private$valid <- TRUE 
            
           #}, warning = function(w) print(w), error = function(e) print(e)) #trycatch
        } #if
      }, #fit
        
        ##
         # 
         ##
         ### 
        predict = function(x)
        {
          #node2vec
          #border_nodes <- as.vector(as.numeric(rownames(x)))
          #idx_app_border = match(border_nodes, private$emb_matrix[,1])
          #x <- cbind(x, private$emb_matrix[idx_app_border, -c(1)])
          #--#
          
          yhat <- predict(private$model$glmnet.fit, x, s = private$model$lambda.min,  type = "response")
          
          names(yhat) <- rownames(x) 
          return (yhat) 

        }
    ),


    private = list(
        
        model = NA,
        model_2 = NA,
        gray_interval = NA,
        emb_matrix = NA, 
        
        heuristicC = NA,
        C = NA,
        bias = NA,
        valid = FALSE
    )
)


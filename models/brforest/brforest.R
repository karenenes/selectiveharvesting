library(R6)

library(randomForest)
library(MLmetrics)

source('node2vec/node-locality.R')
#source('nbne/node-locality.R')


##
 #
 ##
BRForest <- R6Class("BRForest",

    public = list(
    
        ##
         # Initializes the model instance.
         ##
        initialize = function(ntree=500,mtry=NA)
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
          name <- sprintf("BRForest, ntree=%d", private$ntree)
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
          if (length(levels(y)) > 1 && min(table(y))>2) {
              
              #node2vec
              #nodes <- as.vector(as.numeric(rownames(x)))
              #private$emb_matrix <- NA
              #private$emb_matrix <- get_node_locality(nodes_to_charge = nodes, problemName = dataName)
              #idx_app = match(nodes, private$emb_matrix[,1])
              #x <- cbind(x, private$emb_matrix[idx_app, -c(1)])
              #--#
              
              y0 = sample(which(y == 0))
              y1 = sample(which(y == 1))
              
              #split training and validation sets
              idx_training = c(y0[1:round(0.8*length(y0))], y1[1:round(0.8*length(y1))])        
              idx_val = c(y0[round(0.8*length(y0)+1):round(length(y0))], y1[round(0.8*length(y1)+1):round(length(y1))])        
              
              #training set
              y_t = y[idx_training]
              x_t = as.matrix(x[idx_training, ])
              
              y0_t = sample(which(y_t == 0))
              y1_t = sample(which(y_t == 1))
              
              #validation set
              x_val = as.matrix(x[idx_val, ])
              y_val = y[idx_val]
              
              
              #1 - Balancing by voting rule.
              prevalence_rare <- min((length(y0_t)/length(y_t)),(length(y1_t)/length(y_t)))
              
              model1 = randomForest(x = x_t, y = y_t, ntree=private$ntree, mtry=private$mtry, 
                                           cutoff = c(prevalence_rare, 1 - prevalence_rare))
              #2 - Balancing by stratified sample.
              size_rare <- min(length(y0_t), length(y1_t))
              
              model2 = randomForest(x = x_t, y = y_t, ntree=private$ntree, mtry=private$mtry, 
                                           strata = y_t, sampsize = c(size_rare, size_rare))
              #3 - Balancing by class-weight during training.
              model3 = randomForest(x = x_t, y = y_t, ntree=private$ntree, mtry=private$mtry, 
                                    classwt = c(100*(length(y0_t)/length(y_t)), 0.01*(length(y1_t)/length(y_t))))
       
              browser()
              
               
              ## minimising Log Loss gives greater accuracy for the classifier.
              p_val1 = as.numeric(as.vector(predict(model1, x_val, type = "prob")[,"1"]))
              error_vec <- LogLoss(p_val1, as.numeric(as.vector(y_val)))
                
              p_val2 =  as.numeric(as.vector(predict(model2, x_val, type = "prob")[,"1"])) 
              error_vec <- c(error_vec, LogLoss(p_val2, as.numeric(as.vector(y_val)))) 
                
              p_val3 = as.numeric(as.vector(predict(model3, x_val, type = "prob")[,"1"]))
              error_vec <- c(error_vec, LogLoss(p_val3, as.numeric(as.vector(y_val))))
              
              #private$model <- paste("model", which.min(error_vec), sep="")
              
              private$valid = TRUE
          }

        },
        
        ##
         # 
         ##
        predict = function(x)
        {
          #node2vec
          #border_nodes <- as.vector(as.numeric(rownames(x)))
          #idx_app_border = match(border_nodes, private$emb_matrix[,1])
          #x <- cbind(x, private$emb_matrix[idx_app_border, -c(1)])
          #--#
          
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
        emb_matrix = NA, 
        mtry = NA,
        ntree = NA,
        valid = FALSE
    )
)

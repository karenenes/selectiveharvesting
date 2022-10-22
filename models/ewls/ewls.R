# -- karen

library(R6)
#library(glmnet)
library(adagio)
library(BBmisc)

source('models/ewls/internal.R')
#source('node2vec/node-locality.R')
source('tane/tane-locality.R')
#source('tpine/tpine-locality.R')

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
              nodes <- as.vector(as.numeric(colnames(x)))
              private$emb_matrix <- NA
              
              #gerar embeddings
              #private$emb_matrix <- get_node_locality(nodes_to_charge = nodes, problemName = dataName)
              private$emb_matrix <- get_tane_locality(nodes_to_charge = nodes, problemName = dataName, algorithm_type = 'ComplModel')
              #private$emb_matrix <- get_tpine_locality(nodes_to_charge = nodes, problemName = dataName)
              
              idx_app = match(nodes, private$emb_matrix[,1])
              x <- rbind(x, t(private$emb_matrix[idx_app, -c(1)]))
              #--#
              
              y0 = sample(which(y == 0))
              y1 = sample(which(y == 1))
              
              idx_training = c(y0[1:round(0.8*length(y0))], y1[1:round(0.8*length(y1))])        
              idx_val = c(y0[round(0.8*length(y0)+1):round(length(y0))], y1[round(0.8*length(y1)+1):round(length(y1))])        
              
              if (length(levels(y.factor[idx_training])) > 1 && min(table(y.factor[idx_training]))>2) {
              
                #training set
                y_t = y[idx_training]
                x_t = as.matrix(x[, idx_training])
                #validation set
                x_val = as.matrix(x[, idx_val])
                y_val = y[idx_val]
                
                private$model <- InternalEWLS$new(as.matrix(x_t), y_t, beta=private$beta, lambda=private$lambda)
                
                weights_2 = (y_t - as.vector(private$model$predict(x_t)[,1]))^2
                weights_temp <- weights_2/sum(weights_2)
                  
                canmodel_2 = F
                i = 0
                while(canmodel_2 == F && i != 10000){
                  #replace = T
                  sample_idx_2 = sample(1:length(y_t), length(y_t), replace = T, prob = weights_temp)
                  y_t2 = y_t[sample_idx_2]
                  y_aux = sort(y_t2)
                  i = i + 1
                  if(y_aux[3] == 0 && y_aux[length(y_aux)-2] == 1){
                    canmodel_2 = T
                    x_t2 = x_t[, order(y_t2)]
                    y_t2 = y_t2[order(y_t2)]
                  }
                }
                  
                if(i < 10000){
                  private$model_2 <- InternalEWLS$new(as.matrix(x_t2), y_t2, beta=private$beta, lambda=private$lambda)
                  
                  p_val1 <- private$model$predict(x_val)[,1]
                  erro_val1 = as.vector((y_val - p_val1)^2) 
                  
                  p_val2 <- private$model_2$predict(x_val)[,1]
                  erro_val2 = as.vector((y_val - p_val2)^2)
                  
                  #maxsum
                  aux = (erro_val1 - erro_val2) - 0.0000000001
                  aux = aux/(abs(aux))
                
                  idx_sorted = order(p_val1) 
                  p_val_s = p_val1[idx_sorted]
                  aux = aux[idx_sorted] 
                 
                  max_subseq = maxsub(aux, inds = TRUE, compiled = TRUE) 
                  
                  p = 0
                  
                  # calculates de probability of a randomly generated vector of 
                  # s = max_subseq$sum
                  # t = (max_subseq$inds[2] - max_subseq$inds[1]) + 1
                  # for(i in s:t){
                  #   #binomial probability
                  #   p = p + choose(t, i)*(0.5^(i))*(0.5^(t-i))
                  # }
                  
                  if(max_subseq$sum > 1 && p < 0.10){
                   private$gray_interval <- p_val_s[max_subseq$inds]
                  }else{
                   private$gray_interval <- NA  
                   private$model_2 <- NA
                  }
              }  
              private$valid <- TRUE 
              
            }
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
        border_nodes <- as.vector(as.numeric(colnames(x)))
        idx_app_border = match(border_nodes, private$emb_matrix[,1])
        x <- rbind(x, t(private$emb_matrix[idx_app_border, -c(1)]))
        #--#
        
        yhat <- private$model$predict(x)[,1]
        
        if(sum(!is.na(private$gray_interval))!=0){
          gray_inds = (yhat >= private$gray_interval[1]) & (yhat <= private$gray_interval[2]); 
          
          if(any(gray_inds)) {
            x_test = x[,gray_inds,drop=F];
            yhat_2 <- private$model_2$predict(x_test)[,1]
            
            yhat[gray_inds] = yhat_2
          }                  
        }
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


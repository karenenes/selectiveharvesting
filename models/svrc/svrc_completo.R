# -- karen

library(R6)
library(glmnet)
library(adagio)
library(e1071)
library(BBmisc)

source('node2vec/node-locality.R')
#source('nbne/node-locality.R')

##
 #
 ##
SVRC <- R6Class("SVRC",

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
          name <- sprintf("Support Vector Machine Class")
          return(name)

        },
        #getName = function()
        #{
        #  name <- sprintf("Generalized linear model, alpha=%s", 
        #                  as.character(private$alpha))
        #  return (name)
        #},
        
        ##
         #
         ##
        fit = function(x, y){  
        
          y.factor = as.factor(y)
          if (length(levels(y.factor)) > 1 && min(table(y.factor))>2) {
              #result = tryCatch( {
              
              #node2vec
              nodes <- as.vector(as.numeric(rownames(x)))
              private$emb_matrix <- NA
              private$emb_matrix <- get_node_locality(nodes_to_charge = nodes, problemName = dataName)
              idx_app = match(nodes, private$emb_matrix[,1])
              x <- cbind(x, private$emb_matrix[idx_app, -c(1)])
              #--#
            
              #x = normalize(x, method = "range", range = c(0,1), margin = 2, on.constant = "quiet")
              #x = normalize(x, method = "standardize", margin = 2, on.constant = "quiet")
            
              y0 = sample(which(y == 0))
              y1 = sample(which(y == 1))
              
              idx_training = c(y0[1:round(0.8*length(y0))], y1[1:round(0.8*length(y1))])        
              idx_val = c(y0[round(0.8*length(y0)+1):round(length(y0))], y1[round(0.8*length(y1)+1):round(length(y1))])        
              
              #idx_training = c(y0[1:round(0.7*length(y0))], y1[1:round(0.7*length(y1))])        
              #idx_val = c(y0[round(0.7*length(y0)+1):round(length(y0))], y1[round(0.7*length(y1)+1):round(length(y1))])        
              
              #idx_training = c(y0[1:round(0.6*length(y0))], y1[1:round(0.6*length(y1))])        
              #idx_val = c(y0[round(0.6*length(y0)+1):round(length(y0))], y1[round(0.6*length(y1)+1):round(length(y1))])        
              
              #idx_training = c(y0[1:round(0.5*length(y0))], y1[1:round(0.5*length(y1))])        
              #idx_val = c(y0[round(0.5*length(y0)+1):round(length(y0))], y1[round(0.5*length(y1)+1):round(length(y1))])        
              
              if (length(levels(y.factor[idx_training])) > 1 && min(table(y.factor[idx_training]))>2) {
              
                #training set
                y_t = y[idx_training]
                x_t = as.matrix(x[idx_training, ])
                #validation set
                x_val = as.matrix(x[idx_val, ])
                y_val = y[idx_val]
                
                
                fit_1 <- tune.svm(x_t, y_t, gamma = c(0.001,0.01,0.1, 1), tunecontrol = tune.control(sampling = "fix", fix = 85/100))
                #fit_1 <- tune.svm(x_t, y_t, gamma = c(0.001,0.01,0.1, 1), tunecontrol = tune.control(cross = 3))
                private$model <- svm(x = x_t, y = y_t, type = "C-classification", gamma = as.numeric(fit_1$best.parameters), probability = TRUE)
               
              
                pred_treino <- as.data.frame(attr(predict(private$model, x_t, probability = TRUE),which = "probabilities"))
                weights_2 = as.vector((y_t - pred_treino$`1`)^2)

                weights_temp <- weights_2/sum(weights_2)

                roladmodelar_2 = F
                i = 0
                while(roladmodelar_2 == F && i != 10000){
                   sample_idx_2 = sample(1:length(y_t), length(y_t), replace = T, prob = weights_temp)
                   #sample_idx_2 = sample(1:length(y_t), round(0.8*length(y_t)), replace = F, prob = weights_temp)
                   y_t2 = y_t[sample_idx_2]
                   y_aux = sort(y_t2)
                   i = i + 1
                   if(y_aux[3] == 0 && y_aux[length(y_aux)-2] == 1){
                     roladmodelar_2 = T
                     x_t2 = x_t[order(y_t2), ]
                     y_t2 = y_t2[order(y_t2)]
                   }
                }
               
                if(i < 10000){
                   fit_2 <- tune.svm(x = x_t2, y = y_t2, gamma = c(0.001,0.01,0.1, 1), tunecontrol = tune.control(sampling = "fix", fix = 90/100))
                   #private$model_2 <- svm(x = x_t2, y = y_t2, type = "C-classification", gamma = fit_2$best.parameters, probability = TRUE)
                   private$model_2 <- svm(x = x_t2, y = y_t2, type = "C-classification", gamma = as.numeric(fit_2$best.parameters), probability = TRUE)

                   p_val1 <- as.data.frame(attr(predict(private$model, x_val, probability = TRUE),which = "probabilities"))
                   erro_val1 = as.vector((y_val - p_val1$`1`)^2)

                   p_val2 <- as.data.frame(attr(predict(private$model_2, x_val, probability = TRUE),which = "probabilities"))
                   erro_val2 = as.vector((y_val - p_val2$`1`)^2)

                   #maxsum
                   aux = (erro_val1 - erro_val2) - 0.0000000001
                   aux = aux/(abs(aux))

                   idx_sorted = order(p_val1$`1`)
                   p_val_s = p_val1$`1`[idx_sorted]
                   aux = aux[idx_sorted]

                   max_subseq = maxsub(aux, inds = TRUE, compiled = TRUE)
                   
                   p = 0
                   #s = max_subseq$sum
                   #t = (max_subseq$inds[2] - max_subseq$inds[1]) + 1
                   #for(i in s:t){
                   #   p = p + choose(t, i)*(0.5^(i))*(0.5^(t-i))
                   #}
                   
                   
                   if(max_subseq$sum > 1 && p < 0.05){
                   #if(max_subseq$sum > 1 && p < 0.10){
                   #if(max_subseq$sum > 1 && p < 0.15){
                    private$gray_interval <- p_val_s[max_subseq$inds]
                    selected_models[[1]] <<- c(selected_models[[1]], 2)
                   }else{
                    private$gray_interval <- NA
                    private$model_2 <- NA
                    selected_models[[1]] <<- c(selected_models[[1]], 1)
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
        predict = function(x)
        {
          
          #node2vec
          border_nodes <- as.vector(as.numeric(rownames(x)))
          idx_app_border = match(border_nodes, private$emb_matrix[,1])
          x <- cbind(x, private$emb_matrix[idx_app_border, -c(1)])
          #--#
          
          #k <- rownames(x)
          #x = normalize(x, method = "range", range = c(0,1), margin = 2, on.constant = "quiet")
          #x = normalize(x, method = "standardize", margin = 2, on.constant = "quiet")
          #rownames(x) <- k
          
          yhat <- as.data.frame(attr(predict(private$model, x, probability = TRUE),which = "probabilities"))  
          yhat <- yhat$`1`
          
          meiuca_size[[1]] <<- c(meiuca_size[[1]], nrow(x))

          ylabel = c(rep(0, nrow(x)))
          ylabel[match(as.numeric(targets), as.numeric(rownames(x)))] <- 1
          
          #browser()
          
          if(sum(!is.na(private$gray_interval))!=0){
	          gray_inds = (yhat >= private$gray_interval[1]) & (yhat <= private$gray_interval[2]);

            meiuca_interval[[1]] <<- c(meiuca_interval[[1]], private$gray_interval[1])
            meiuca_interval[[2]] <<- c(meiuca_interval[[2]], private$gray_interval[2])

            if(any(gray_inds)) {
              x_test = x[gray_inds,,drop=F];
              meiuca_size[[2]] <<- c(meiuca_size[[2]], nrow(x_test))

              yhat_2 <- as.data.frame(attr(predict(private$model_2, x_test, probability = TRUE),which = "probabilities"))
              yhat_2 <- yhat_2$`1`

              recruited_node_model[[1]] <<- c(recruited_node_model[[1]], max(max(yhat[-which(gray_inds == T)]), max(yhat_2)))

              if(max(yhat[-which(gray_inds == T)]) > max(yhat_2)){
                recruited_node_model[[2]] <<- c(recruited_node_model[[2]], 1)
              }else{
                recruited_node_model[[2]] <<- c(recruited_node_model[[2]], 2)
              }

              error_yhat <<- c(error_yhat, mean((ylabel[gray_inds] - yhat[gray_inds])^2))
              error_yhat_2 <<- c(error_yhat_2, mean((ylabel[gray_inds] - as.vector(yhat_2))^2))

              yhat[gray_inds] = yhat_2

              selected_models[[2]] <<- c(selected_models[[2]], 2)
              }else{
               meiuca_size[[2]] <<- c(meiuca_size[[2]], -1)
               selected_models[[2]] <<- c(selected_models[[2]], 1)
              }
          }else{
            meiuca_interval[[1]] <<- c(meiuca_interval[[1]], -1)
            meiuca_interval[[2]] <<- c(meiuca_interval[[2]], -1)
            meiuca_size[[2]] <<- c(meiuca_size[[2]], -1)
            selected_models[[2]] <<- c(selected_models[[2]], 1)

	        }
          
          #browser()
          
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


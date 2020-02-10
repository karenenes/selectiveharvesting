# -- karen

library(R6)
library(glmnet)
library(adagio)
library(BBmisc)
library(naivebayes)

source('node2vec/node-locality.R')
#source('nbne/node-locality.R')

NB <- R6Class("NB",
               
               public = list(
                 
                 ##
                 # Initializes the model instance.
                 # Default parameter values are those from glmnet.
                 ##
                 #},
                 
                 initialize = function() {},
                 
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
                   name <- sprintf("Naive Bayes")
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
                     
                     private$model <- naive_bayes(x = x, y = y)
                     
                     browser()
                     
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
                   
                   yhat <- as.data.frame(predict(private$model, x, type = "prob"))
                   yhat <- yhat$`1`
                   
                   names(yhat) <- rownames(x) 
                   return (yhat) 
                   
                 }
               ),
               
               
               private = list(
                 
                 model = NA,
                 emb_matrix = NA, 
                 
                 valid = FALSE
               )
)

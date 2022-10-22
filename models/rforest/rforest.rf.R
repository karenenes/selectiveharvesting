library(R6)

library(randomForest)

#source('node2vec/node-locality.R')
#source('tane/tane-locality.R')
#source('tpine/tpine-locality.R')


##
 #
 ##
RForest.rf <- R6Class("RForest.rf",

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
          name <- sprintf("RForest.rf, ntree=%d", private$ntree)
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

          y.factor = as.factor(y)
          if (length(levels(y.factor)) > 1 && min(table(y.factor))>2) { ###karen
          #if (length(levels(y)) > 1) { ###original
              y = as.factor(y)
              
              ###
              Nturn <<- Nturn + 1 # snapshots -- Karen  
              
              #node2vec
              #nodes <- as.vector(as.numeric(colnames(x)))
              #private$emb_matrix <- NA
              
              #gerar embeddings
              #private$emb_matrix <- get_node_locality(nodes_to_charge = nodes, problemName = dataName)
              #private$emb_matrix <- get_tane_locality(nodes_to_charge = nodes, problemName = dataName, algorithm_type = 'D3ts')
              #private$emb_matrix <- get_tpine_locality(nodes_to_charge = nodes, problemName = dataName)
              
              #idx_app = match(nodes, private$emb_matrix[,1])
              #x <- rbind(x, t(private$emb_matrix[idx_app, -c(1)]))
              #--#
              
              #browser()
              private$model = randomForest(x, y, ntree=private$ntree, mtry=private$mtry)
              #feature_importance <<- c(feature_importance, list(private$model$importance)) # -- karen
              
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

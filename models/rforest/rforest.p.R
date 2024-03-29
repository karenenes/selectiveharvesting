library(R6)

library(party)

#source('node2vec/node-locality.R')
source('tane/tane-locality.R')
#source('tpine/tpine-locality.R')


##
 #
 ##
RForest.p <- R6Class("RForest.p",

    public = list(
    
        ##
         # Initializes the model instance.
         ##
        initialize = function(ntree=500)
        {
          private$ntree <- ntree
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
          name <- sprintf("RForest.p, ntree=%d", private$ntree)
          return (name)
        },
        
        ##
         #
         ##
        fit = function(x, y)
        {
          if (is.na(private$mtry))
            private$mtry = ceiling(sqrt(ncol(x)))

          y.factor = as.factor(y)
          
          if (length(levels(y.factor)) > 1 && min(table(y.factor))>2) { ### karen
          #if (length(levels(y)) > 1) {  ## original
              y = as.factor(y)
              
              #node2vec
              #nodes <- as.vector(as.numeric(rownames(x)))
              #private$emb_matrix <- NA
              #private$emb_matrix <- get_node_locality(nodes_to_charge = nodes, problemName = dataName)
              #idx_app = match(nodes, private$emb_matrix[,1])
              #x <- cbind(x, private$emb_matrix[idx_app, -c(1)])   
              #--#
             
              private$model = cforest( y ~ ., data.frame(x,y), control=cforest_classical(ntree=private$ntree,mtry=private$mtry) )
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
              yhat <- unlist(treeresponse( private$model, newdata=data.frame(x)))[2*(1:nrow(x))]
          } else {
              yhat <- rep(0,nrow(x))
          }
          names(yhat) <- rownames(x)
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

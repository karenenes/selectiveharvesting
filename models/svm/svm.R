library(R6)

library(LiblineaR)

#source('node2vec/node-locality.R')

##
 #
 ##
SVM <- R6Class("SVM",

    public = list(
    
        ##
         # Initializes the model instance.
         # Default parameter values are those from LiblineaR.
         ##
        initialize = function(C=1, bias=TRUE)
        {
          private$heuristicC <- (C == "heuristic")
          private$C <- C
          private$bias <- bias
          private$valid <- FALSE
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
          name <- sprintf("SVM, C=%s, bias=%s", 
                          as.character(private$C),
                          as.character(private$bias))
          return (name)
        },
        
        ##
         #
         ##
        fit = function(x, y)
        {
          
          y.factor = as.factor(y)
          if (length(levels(y.factor)) > 1 && min(table(y.factor))>2  ) {
            y = as.factor(y)  
            #node2vec
            #nodes <- as.vector(as.numeric(rownames(x)))
            #private$emb_matrix <- NA
            #private$emb_matrix <- get_node_locality(nodes_to_charge = nodes, problemName = dataName)
            #idx_app = match(nodes, private$emb_matrix[,1])
            #x <- cbind(x, private$emb_matrix[idx_app, -c(1)])
            #--#
        
            cost <- if (private$heuristicC) heuristicC(x) else private$C
            
            if(!is.infinite(cost)){
                private$model <- LiblineaR(data=x, target=y, type=11, svr_eps=0.1, bias=private$bias, cost=cost)
                private$valid <- TRUE
            }else
            {
                #browser()
                private$valid <- FALSE
            }
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
          
          yhat <- predict(private$model, x)$predictions
          names(yhat) <- rownames(x)
          return (yhat)
        }
    ),

    private = list(
        
        model = NA,
        valid = NA,
        emb_matrix = NA, 
        heuristicC = NA,
        C = NA,
        bias = NA
    )
)

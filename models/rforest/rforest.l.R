library(R6)

library(party)

##
 #
 ##
RForest.l <- R6Class("RForest.l",

    public = list(
    
        ##
         # Initializes the model instance.
         ##
        initialize = function(ntree=500, pickOne=FALSE)
        {
          private$ntree <- ntree
          private$pickOne <- pickOne
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
          name <- sprintf("RForest.l, ntree=%d, pickOne=%d", private$ntree, 1*private$pickOne)
          return (name)
        },
        
        ##
         #
         ##
        fit = function(x, y)
        {
          if (is.na(private$mtry))
            private$mtry = ceiling((ncol(x)**(2/3)))

          y = as.factor(y)
          if (length(levels(y)) > 1) {
              private$model = cforest( y ~ ., data.frame(x,y), control=cforest_classical(ntree=private$ntree,mtry=private$mtry) )
              private$valid = TRUE
          }

        },

        #id2prediction = function(root, id) {
        #  if (root[[1]] == id) {
        #    return (root[[7]])
        #  } else if (root[[4]] == FALSE) {
        #    found = self$id2prediction(root[[8]], id)
        #    if (!is.null(found)) {
        #      return (found)
        #    } else {
        #      return (self$id2prediction(root[[9]], id))
        #    }
        #  }
        #  return (NULL)
        #},

        ##
         # 
         ##
        point2prediction = function(point, root) {
          if (root[[4]] == TRUE) { # terminal node
            return (root[[7]])     # prediction
          } else {
            feat_ind = root[[5]][[1]]
            threshold = root[[5]][[3]]
            if ( point[feat_ind] < threshold ) {
              return (self$point2prediction(point, root[[8]]) )   # left subtree
            } else {
              return  (self$point2prediction(point, root[[9]]) )  # right subtree
            }
          }
          return (NULL)
        },
        
        ##
         # 
         ##
        predict = function(x)
        {
          if (private$valid) {
            if(!private$pickOne) {
              yhat <- unlist(treeresponse(private$model, newdata=data.frame(x)))[2*(1:nrow(x))]
            } else {
              ensemble <- private$model@ensemble
              tree.ind <- sample.int(length(ensemble), size=1)
              yhat <- apply(x,1,self$point2prediction,ensemble[[tree.ind]])[2,]
            }
          } else {
              yhat <- rep(0,nrow(x))
          }
          names(yhat) <- rownames(x)
          return(yhat)
        }
    ),

    private = list(
        
        model = NA,
        mtry = NA,
        pickOne = FALSE,
        ntree = NA,
        valid = FALSE
    )
)

library(R6)
library(magrittr)

source('oracle/node-oracle.R')
##
 # General model class.
 # Encapsulates specific model implementations.
 ##
Model <- R6Class("Model",

    public = list(
    
        ##
         # Initializes the model and general settings.
         ##
        initialize = function(settings, model, observed_graph, external_feats)
        {
          private$settings <- settings
          private$model <- model$clone()
          private$type <- class(model)[1]
          private$graph <- observed_graph
          private$must_fit <- FALSE
          private$external_feats <- external_feats
        },
        
        ##
         # Fits a new model on the current samples or updates it with the new sample.
         ##
        recruitNode = function(id)
        {
          if (length(private$graph$recruited) > 1) {
            private$update(id)
          } else {
            private$fit()
          }
        },
        
        ##
         # Returns TRUE if the model is valid (can be used for prediction).
         # Returns FALSE otherwise.
         ##
        isValid = function()
        {
          if ((!private$model$isValid()) && private$must_fit) private$fit()
          return (private$model$isValid())
        },
        
        ##
         # Returns a character string identifying the underlying model.
         ##
        getName = function()
        {
          return (private$model$getName())
        },
        
        ##
         # 
         ##
        evaluateBorder = function()
        {
          return (private$evaluateNodes(private$graph$border))
        },
        
        ##
         # Allows internal access to the instance for testing/debugging purposes.
         ##
        browse = function()
        {
          browser()
        }
    ),

    private = list(
        
        model = NA,           # A reference to an initialized instance of a supported model class.
        type = NA,            # (character) Name of the R6 class which implements private$model.
        settings = NA,        # (Settings) General settings.
        graph = NA,           # (ObservedGraph) Reference to the currently observed graph.
        must_fit = NA,        # (logical) Indicates whether fit() must be called prior to predicting.
        external_feats = NA,  # (matrix) Rows 1:i compose external design matrix at turn i.
        oracle_embedding = NA, # oracle embedding matrix
        
        # Supported models. Each constant value must be the name of the corresponding R6 class.
        TYPE_SVM = "SVM",
        TYPE_EWRLS = "EWRLS",
        TYPE_EWLS = "EWLS",
        TYPE_MFOREST = "MForest",
        TYPE_RFOREST_RF = "RForest.rf",
        TYPE_RFOREST_P = "RForest.p",
        TYPE_RFOREST_L = "RForest.l",
        TYPE_LISTNET = "ListNet",
        TYPE_ACTIVESEARCH = "ActiveSearch",
        TYPE_SIGMAOPT = "SigmaOpt",
        TYPE_LOGISTIC = "Logistic",
        TYPE_MOD = "MOD",
        TYPE_ADABOOST = "AdaBoost",
        TYPE_PYBOOSTING = "PyBoosting",
        TYPE_BAGGING = "Bagging",
        TYPE_RANDOM = "Random",
        TYPE_GLM = "GLM",
        TYPE_SVR = "SVR",
        TYPE_SVRC = "SVRC",
        ##
         #
         ##
        fit = function()
        {
          # Unsets the lazy fit flag.
          private$must_fit <- FALSE

          # Gets training data from recruited nodes or the external features matrix. 
          N = length(private$graph$recruited)
          if(!is.null(private$external_feats)) {
            x = as.matrix(private$external_feats[1:N, -1, drop=FALSE])
            y = as.vector(private$external_feats[1:N, "response", drop=FALSE])
          } else {
            x = as.matrix(private$graph$features[private$graph$recruited, -1, drop=FALSE])
            y = as.vector(private$graph$features[private$graph$recruited, "response"])
          }
          
          # EWRLS.
          if (private$type == private$TYPE_EWRLS) {
            private$model$fit(
                x = t(cbind(x, rep(1, N))),
                y = y)
          }
          
          
          # EWLS.
          if (private$type == private$TYPE_EWLS) {
            ##get oracle
            #nodes <- as.vector(as.numeric(rownames(x)))
            #private$oracle_embedding <- get_oracle(nodes_to_charge = nodes, problemName = dataName)
            #x <- cbind(x, private$oracle_embedding)
            ##--
            private$model$fit(
              x = t(cbind(x, rep(1, N))),
              y = y)
          }

          # ListNet.
          if (private$type == private$TYPE_LISTNET) {
            private$model$fit(
                x = cbind(x, rep(1, N)),
                y = y)
          }

          # ActiveSearch.
          if (private$type %in% c(private$TYPE_ACTIVESEARCH,private$TYPE_SIGMAOPT)) {}

          # PyBoosting.
          if (private$type == private$TYPE_PYBOOSTING) {
            private$model$fit(
                x = cbind(x, rep(1, N)),
                y = y)
          }
          
          # SVM.
          if (private$type == private$TYPE_SVM) {
            ##get oracle
            #nodes <- as.vector(as.numeric(rownames(x)))
            #private$oracle_embedding <- get_oracle(nodes_to_charge = nodes, problemName = dataName)
            #x <- cbind(x, private$oracle_embedding)
            ##--
            private$model$fit(
                x = x,
                y = y)
          }

          # Logistic.
          if (private$type == private$TYPE_LOGISTIC) {
            private$model$fit(
                x = x,
                y = y)
          }
          
          # RForest.
          if (private$type == private$TYPE_RFOREST_RF || 
              private$type == private$TYPE_RFOREST_P || 
              private$type == private$TYPE_RFOREST_L) {
            ##get oracle
            #nodes <- as.vector(as.numeric(rownames(x)))
            #private$oracle_embedding <- get_oracle(nodes_to_charge = nodes, problemName = dataName)
            #x <- cbind(x, private$oracle_embedding)
            ##--
            private$model$fit(
                x = x,
                y = y)
          }
          
          # MForest.
          if (private$type == private$TYPE_MFOREST) {
            private$model$fit(
                x = x,
                y = y)
          }
          
          # MOD.
          if (private$type == private$TYPE_MOD) {}

          # AdaBoost.
          if (private$type == private$TYPE_ADABOOST || private$type == private$TYPE_BAGGING) {
            private$model$fit(
                x = x,
                y = y)
          }
          
          # Random.
          if (private$type == private$TYPE_RANDOM) {}

          # GLM
          if (private$type == private$TYPE_GLM) {
            ##get oracle
            #nodes <- as.vector(as.numeric(rownames(x)))
            #private$oracle_embedding <- get_oracle(nodes_to_charge = nodes, problemName = dataName)
            #x <- cbind(x, private$oracle_embedding)
            ##--
            private$model$fit(
                x = x,
                y = y)
          }
          
          # SVR
          if (private$type == private$TYPE_SVR) {
            ##get oracle
            #nodes <- as.vector(as.numeric(rownames(x)))
            #private$oracle_embedding <- get_oracle(nodes_to_charge = nodes, problemName = dataName)
            #x <- cbind(x, private$oracle_embedding)
            ##--
            private$model$fit(
              x = x,
              y = y)
          }

          # SVRC
          if (private$type == private$TYPE_SVRC) {
            ##get oracle
            #nodes <- as.vector(as.numeric(rownames(x)))
            #private$oracle_embedding <- get_oracle(nodes_to_charge = nodes, problemName = dataName)
            #x <- cbind(x, private$oracle_embedding)
            ##--
            private$model$fit(
              x = x,
              y = y)
          }
        },

        ##
         #
         ##
        update = function(id)
        {
          stopifnot(id %in% private$graph$recruited)

          # Gets node data from recruited nodes or the external features matrix.
          N = length(private$graph$recruited)
          if(!is.null(private$external_feats)) {
            x = private$external_feats[N, -1]
            y = private$external_feats[N, "response"]
          } else {
            x = private$graph$features[id, -1]
            y = private$graph$features[id, "response"]
          }
          
          # EWRLS.
          if (private$type == private$TYPE_EWRLS) {
            private$model$update(x=c(x, 1), y=y)
          }
          
          # EWLS.
          if (private$type == private$TYPE_EWLS) {
            #private$model$update(x=c(x, 1), y=y)
            private$flagLazyFit()
          }

          # ListNet.
          if (private$type == private$TYPE_LISTNET) {
            private$flagLazyFit()
          }

          # ActiveSearch.
          if (private$type %in% c(private$TYPE_ACTIVESEARCH,private$TYPE_SIGMAOPT)) {}

          # PyBoosting.
          if (private$type == private$TYPE_PYBOOSTING) {
            private$flagLazyFit()
          }
          
          # SVM.
          if (private$type == private$TYPE_SVM) {
            private$flagLazyFit()
          }

          # Logistic.
          if (private$type == private$TYPE_LOGISTIC) {
            private$flagLazyFit()
          }
          
          # MOD.
          if (private$type == private$TYPE_MOD) {}

          # AdaBoost.
          if (private$type == private$TYPE_ADABOOST || private$type == private$TYPE_BAGGING) {
            private$flagLazyFit()
          }

          # RForest.
          if (private$type == private$TYPE_RFOREST_RF || 
              private$type == private$TYPE_RFOREST_P || 
              private$type == private$TYPE_RFOREST_L) {
            private$flagLazyFit()
          }
          
          # MForest.
          if (private$type == private$TYPE_MFOREST) {
            if (self$isValid()) {
              private$model$update(x=x, y=y)
            } else {
              private$fit()
            }
          }
          
          # Random.
          if (private$type == private$TYPE_RANDOM) {}

          # GLM
          if (private$type == private$TYPE_GLM) {
            private$flagLazyFit()
          }
          
          # SVR
          if (private$type == private$TYPE_SVR) {
            private$flagLazyFit()
          }
          
          # SVRC
          if (private$type == private$TYPE_SVRC) {
            private$flagLazyFit()
          }
        },
        
        ##
         #
         ##
        predict = function(ids)
        {
          # Refits the model if flagged.
          if (private$must_fit) {
            private$fit()
          }
          
          # EWRLS.
          if (private$type == private$TYPE_EWRLS) {
            
            return (private$model$predict(x = t(private$graph$getDesignMatrix(ids))))
          }
          
          # EWLS.
          if (private$type == private$TYPE_EWLS) {
            x = private$graph$getDesignMatrix(ids)
            ##get oracle
            #nodes <- as.vector(as.numeric(rownames(x)))
            #private$oracle_embedding <- get_oracle(nodes_to_charge = nodes, problemName = dataName)
            #x <- cbind(x, private$oracle_embedding)
            ##--
            x = t(x)
            return (private$model$predict(x = x))
          }

          # ListNet.
          if (private$type == private$TYPE_LISTNET) {
            return (private$model$predict(x = private$graph$getDesignMatrix(ids))[,1])
          }

          # ActiveSearch.
          if (private$type %in% c(private$TYPE_ACTIVESEARCH, private$TYPE_SIGMAOPT)) {
            params <- private$graph$getAdjacencyMatrix()
            # Builds two vectors, which will indicate the recruited nodes and their labels.
            isRecruited <- rep(FALSE, length(params$node_ids)) %>% set_names(params$node_ids)
            responses <- isRecruited
            # Sets all recruited nodes and their labels in the parameter vectors, with the exception of those
            # recruited nodes for which we want to predict the label (recruited nodes which are also in ids).
            known_nodes <- setdiff(private$graph$recruited, ids)
            responses[known_nodes] <- private$graph$features[known_nodes, "response"]
            isRecruited[known_nodes] <- TRUE
            # Predicts the labels using the model.
            yhat <- private$model$predict(params$edge_matrix, isRecruited, responses)
            return (yhat[ids])
          }

          # PyBoosting.
          if (private$type == private$TYPE_PYBOOSTING) {
            yhat <- private$model$predict(x = private$graph$getDesignMatrix(ids))
            names(yhat) <- ids
            return (yhat)
          }
          
          
          # SVM.
          if (private$type == private$TYPE_SVM) {
            x = as.matrix(private$graph$features[ids, -1, drop=FALSE])
            ##get oracle
            #nodes <- as.vector(as.numeric(rownames(x)))
            #private$oracle_embedding <- get_oracle(nodes_to_charge = nodes, problemName = dataName)
            #x <- cbind(x, private$oracle_embedding)
            ##--
            
            return (private$model$predict(x = x)) 
          }

          # Logistic.
          if (private$type == private$TYPE_LOGISTIC) {
            return (private$model$predict(x = as.matrix(private$graph$features[ids, -1, drop=FALSE])))
          }

          # RForest.
          if (private$type == private$TYPE_RFOREST_RF || 
              private$type == private$TYPE_RFOREST_P || 
              private$type == private$TYPE_RFOREST_L) {
                  x = as.matrix(private$graph$features[ids, -1, drop=FALSE])
                  ##get oracle
                  #nodes <- as.vector(as.numeric(rownames(x)))
                  #private$oracle_embedding <- get_oracle(nodes_to_charge = nodes, problemName = dataName)
                  #x <- cbind(x, private$oracle_embedding)
                  ##--
                  
                  return (private$model$predict(x = x)) 
          }
          
          # MForest.
          if (private$type == private$TYPE_MFOREST) {
            return (private$model$predict(x = as.matrix(private$graph$features[ids, -1, drop=FALSE])))
          }

          # AdaBoost.
          if (private$type == private$TYPE_ADABOOST || private$type == private$TYPE_BAGGING) {
            return (private$model$predict(x = as.matrix(private$graph$features[ids, -1, drop=FALSE])))
          }
          
          # MOD.
          if (private$type == private$TYPE_MOD) {
            return (private$graph$getPositiveNeighborsCount(ids))
          }
          
          # Random.
          if (private$type == private$TYPE_RANDOM) {
            y <- unlist(Map(function(id) 0, ids))
            y[sample.int(length(y), size=1)] <- 1
            return (y)
          }

          # GLM.
          if (private$type == private$TYPE_GLM) {
            x = as.matrix(private$graph$features[ids, -1, drop=FALSE])
            ##get oracle
            #nodes <- as.vector(as.numeric(rownames(x)))
            #private$oracle_embedding <- get_oracle(nodes_to_charge = nodes, problemName = dataName)
            #x <- cbind(x, private$oracle_embedding)
            ##--:
            
            return (private$model$predict(x = x)) 
          }
          
          # SVR.
          if (private$type == private$TYPE_SVR) {
            x = as.matrix(private$graph$features[ids, -1, drop=FALSE])
            ##get oracle
            #nodes <- as.vector(as.numeric(rownames(x)))
            #private$oracle_embedding <- get_oracle(nodes_to_charge = nodes, problemName = dataName)
            #x <- cbind(x, private$oracle_embedding)
            ##--:
            
            return (private$model$predict(x = x)) 
          }
          
          # SVRC.
          if (private$type == private$TYPE_SVRC) {
            x = as.matrix(private$graph$features[ids, -1, drop=FALSE])
            ##get oracle
            #nodes <- as.vector(as.numeric(rownames(x)))
            #private$oracle_embedding <- get_oracle(nodes_to_charge = nodes, problemName = dataName)
            #x <- cbind(x, private$oracle_embedding)
            ##--:
            
            return (private$model$predict(x = x)) 
          }
        },
        
        ##
         # 
         ##
        evaluateNodes = function(ids)
        {
          stopifnot(self$isValid())
          if (length(ids) == 0) {
            return (NULL)
          }
          return (private$predict(ids))
        },
        
        ##
         # If the model is valid, it is simply flagged to be refitted prior to predicting.
         # If the model is not yet valid, always retrains it.
         ##
        flagLazyFit = function()
        {
          if (self$isValid()) {
            private$must_fit <- TRUE
          } else {
            private$fit()
          }
        }
    )
)

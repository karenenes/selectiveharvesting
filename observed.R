# -- KAREN

library(R6)

library(graph)
library(Matrix)
library(nnet)
library(Pay2Recruit)

##
 #
 ##
ObservedGraph <- R6Class("ObservedGraph",

    public = list(
      
        ##
         # These member variables are public so that they can be directly referenced by the models.
         # They MUST NOT be directly modified outside of this class.
         ##
        features = NULL,    # (Matrix) Computed features of observed nodes and  response values of recruited ones.
        border = c(),       # (vector) Border nodes (observed but not yet recruited).
        recruited = c(),    # (vector) Recruited nodes.
    
        ##
         #
         ##
        initialize = function(settings)
        {
          private$settings <- settings
          
          # Initializes the observed graph (Graph from Pay2Recruit).
          private$graph <- new(Graph, undirected=TRUE)
          private$graph$initFeatures2(ntypes = 2, 
                                      target = settings$problem$accept_trait, 
                                      nattribs = settings$problem$nattribs,
                                      #include_structural = settings$feature_set$include_structural,
                                      include_structural = 0,
                                      include_triangles = settings$feature_set$include_triangles, 
                                      include_most = settings$feature_set$include_most, 
                                      include_rw = settings$feature_set$include_rw, 
                                      include_counts = settings$feature_set$include_counts, 
                                      include_fractions = settings$feature_set$include_fractions, 
                                      include_attrib_fractions = settings$feature_set$include_attrib_fractions)#,
                                      #include_attrib_fractions = 0)
                                      
          
          # Initializes the observed graph (graphNEL from graph, package from Bioconductor).
          private$bc_graph <- graphNEL(nodes=character(), edgeL=list(), edgemode="undirected")
          
          # Initializes the features matrix.
          private$relocateFeaturesMatrix()
        },
        
        ##
         # Recruits a new node.
         ##
        recruitNode = function(id, info, border_check=TRUE)
        {
          # Asserts the node has not been recruited yet and is in the border set.
          stopifnot(!(id %in% self$recruited))
          if (border_check) {
            stopifnot(id %in% self$border)
          }

          # Updates the border and recruited node sets.
          self$recruited <- union(self$recruited, id)
          self$border <- union(self$border, setdiff(info$neighbors, self$recruited))
          self$border <- setdiff(self$border, id)

          # Adds the node's information to the network and updates the features matrix.
          private$addNodeP2R(id, info)
          private$addNodeBC(id, info)
        },
        
        ##
         # Returns the number of positive neighbors of each specified node.
         ##
        getPositiveNeighborsCount = function(ids)
        {
          stopifnot(length(ids) > 0)
          stopifnot(all(ids %in% rownames(self$features)))
          counts <- private$graph$getPositiveNeighborsCount(as.numeric(ids))
          names(counts) <- ids
          return (counts)
        },
        
        ##
         # Returns the number of neighbors of each specified node.
         ##
        getNeighborsCount = function(ids)
        {
          stopifnot(length(ids) > 0)
          stopifnot(all(ids %in% rownames(self$features)))
          counts <- private$graph$getNeighborsCount(as.numeric(ids))
          names(counts) <- ids
          return (counts)
        },
        
        ##
         # Returns the ID of the border node which has the greatest number of positive neighbors.
         ##
        getModSuggestion = function()
        {
          ods <- self$getPositiveNeighborsCount(self$border)
          return (names(ods)[which.is.max(ods)])
        },
        
        ##
         # Returns the ID of a random border node.
         ##
        getRandomSuggestion = function()
        {
          return (sample(self$border, 1))
        },
        
        ##
         # Returns an N by P+1 matrix X where N is length(ids) and P is the number of node features.
         # The last column of X is a vector of ones that accounts for the intercept (bias) in the regression model.
         ##
        getDesignMatrix = function(ids)
        {
          return (cbind(as.matrix(self$features[ids, -1, drop=FALSE]), rep(1, length(ids))))
        },
        
        ##
         # Returns the observed graph as an edge matrix.
         ##
        getAdjacencyMatrix = function()
        {
          return (list(edge_matrix = edgeMatrix(private$bc_graph, duplicates=TRUE),
                       node_ids = graph::nodes(private$bc_graph)))
        },

        ##
         # Returns the number of edges in the observed graph.
         ##
        getNumEdges = function()
        {
          return (numEdges(private$bc_graph))
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
        
        settings = NA,        # (Settings) General settings.
        graph = NA,           # (Graph from Pay2Recruit) The observed graph.
        bc_graph = NA,        # (graphNEL from graph) The observed graph.
        nobserved = 0,        # (integer >= 0) Number of observed nodes.

        ##
         # Adds a node to the graph (Pay2Recruit) and updates the features matrix.
         ##
        addNodeP2R = function(id, info)
        {
          new_feats <- tryCatch({private$graph$addNode4(as.numeric(id), 
                                                        as.numeric(info$neighbors), 
                                                        info$label, 
                                                        info$attributes)},
                                warning = function(w) {print(w); print(id); print(info)},
                                error = function(e) {print(e); print(id); print(info)},
                                finally = {})
          private$updateFeatures(new_feats)
          self$features[id, "response"] <- info$label
        },
        
        ##
         # Adds a node to the graph (graphNEL from graph, package from Bioconductor).
         ##
        addNodeBC = function(id, info)
        {
          new_nodes <- setdiff(union(id, info$neighbors), graph::nodes(private$bc_graph))
          private$bc_graph <- addNode(new_nodes, private$bc_graph)
          private$bc_graph <- suppressWarnings(addEdge(id, info$neighbors, private$bc_graph))
        },
        
        ##
         # Relocates the features to a new, larger matrix, in order to store more node data.
         ##
        relocateFeaturesMatrix = function()
        {
          feature_names <- append("response", private$graph$getFeatureNames())
          new_size <- if (is.null(self$features)) {1000} else {2 * nrow(self$features)}
          new_matrix <- Matrix(0, nrow=new_size, ncol=length(feature_names), sparse=TRUE)
          colnames(new_matrix) <- feature_names
          rownames(new_matrix) <- rep(NA, new_size)
          
          # Copies the old data over to the new matrix.
          if (!is.null(self$features)) {
            new_matrix[1:private$nobserved,] <- self$features[1:private$nobserved,]
            rownames(new_matrix)[1:private$nobserved] <- rownames(self$features)[1:private$nobserved]
          }
          self$features <- new_matrix
        },
        
        ##
         # Updates the features of a group of observed nodes.
         # The method's parameter is the matrix returned by Graph::addNode4().
         ##
        updateFeatures = function(new_feats)
        {
          ids <- as.character(as.integer(new_feats[1,]))
          curr_ids <- rownames(self$features)[1:private$nobserved]
          new_ids <- setdiff(ids, curr_ids)
          
          # If needed, relocates the features matrix.
          nrows <- private$nobserved + length(new_ids)
          while (nrows > nrow(self$features)) {
            private$relocateFeaturesMatrix()
          }

          # Assigns rows from the feature matrix to the new IDs.
          if (length(new_ids) > 0) {
            row_names <- rownames(self$features)
            row_names[(private$nobserved+1):nrows] <- new_ids
            rownames(self$features) <- row_names
          }

          # Updates the feature values.
          private$nobserved <- nrows
          self$features[ids,-1] <- t(new_feats[-1,])
        }
    )
)

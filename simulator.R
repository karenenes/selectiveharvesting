library(R6)

library(nnet)
library(Pay2Recruit)

source("model.R")
source("observed.R")
source("selector.R")

##
 #
 ##
Simulator <- R6Class("Simulator",

    public = list(
    
        ##
         # Initializes a new instance.
         ##
        initialize = function(name=NA, graph=NA, settings=NA, external_feats=NA)
        {
          stopifnot(settings$validate())
          
          private$settings <- settings$clone()
          private$name <- name
          private$graph <- graph
          private$external_feats <- external_feats
        },
        
        ##
         #
         ##
        simulate = function(seeds)
        {
          private$start(seeds)
          stopifnot(private$turn < private$settings$nattempts)
          private$runTurns(1 + private$settings$nattempts - private$turn)
        },
        
        ##
         # Returns results of the simulation at the given turn.
         ##
        getSimulationResultsAtTurn = function(turn)
        {
          stopifnot(turn < private$turn)
          
          payoff <- self$getTotalPayoffAtTurn(turn)
          recruited  <- private$selected_nodes[1:turn]
          nrecruited <- length(recruited)
          features <- if (private$settings$save_observations) private$observed_feats[turn,] else NULL
          
          return (SimulationResult$new(payoff, 
                                       nrecruited, 
                                       payoff / nrecruited, 
                                       recruited[nrecruited],
                                       private$selected_models[turn],
                                       features))
        },
        
        ##
         # Returns the total payoff of nodes recruited up to the given turn.
         ##
        getTotalPayoffAtTurn = function(turn)
        {
          nodes <- private$getRecruitedUpToTurn(turn)
          total <- Reduce(function(acc, node_id) acc + private$recruited[[node_id]]$label, nodes, 0)
          return (total)
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
    
        name = NA,              # (character) Simulation name.
        settings = NA,          # (Settings) Simulation and model settings.
        
        graph = NA,             # (Graph) The full graph (underlying graph).
        observed_graph = NA,    # (ObservedGraph) The currently observed graph.
        
        models = NA,            # (list<Model>) The list of models used during the recruitment process.
        action_selector = NA,   # (ActionSelector) An action selector which encapsulates the configured policy.

        turn = NA,              # (integer >= 0) The simulation turn counter.        
        recruited = NA,         # (list) A list, indexed by node ID, with data about all recruited nodes.
        
        selected_nodes = NA,    # (vector) Vector with IDs of all recruited nodes, in recruitment order.
        selected_models = NA,   # (vector) Vector with the ID of the model selected in each turn.
        
        observed_feats = NA,    # (matrix) Rows 1:i compose design matrix at turn i.
        external_feats = NA,    # (matrix) Rows 1:i compose external design matrix at turn i.
        
        # Class constants.
        NEWLY_RECRUITED = 1,
        EMPTY_BORDER = 3,

        ##
         # Prints a message from this simulator.
         ##
        display = function(msg)
        {
          cat(paste(private$name, ": ", msg, sep=""))
        },

        ##
         # Returns TRUE if the node with the given ID has been recruited.
         # Returns FALSE otherwise.
         ##
        hasRecruited = function(id)
        {
          return (id %in% names(private$recruited))
        },
        
        ##
         #
         ##
        getSimulationState = function()
        {
          return (SimulationState$new(current_turn = private$turn,
                                      final_turn = private$settings$nattempts,
                                      observed_graph = private$observed_graph))
        },

        ##
         # Recruits a new node and updates all models.
         # Returns the node info.
         ##
        recruitNode = function(node_id, border_check=TRUE)
        {
          stopifnot(!private$hasRecruited(node_id))
          
          # Gets the node info.
          node_info <- private$getNodeInfo(node_id)
          
          # Adds the new sample to the list of recruited nodes.
          private$selected_nodes <- c(private$selected_nodes, node_id)
          private$recruited[[node_id]] <- list(turn = private$turn, 
                                               label = node_info$label)
          
          # Adds the new node to the observed graph.
          private$observed_graph$recruitNode(node_id, node_info, border_check)

          # Updates the observed design matrix.
          private$observed_feats <- rBind(private$observed_feats, private$observed_graph$features[node_id,])

          # Updates all models.
          Map(function(i) private$models[[i]]$recruitNode(node_id), 1:private$settings$nmodels)
          
          return (node_info)
        },
        
        ##
         # Returns the IDs of all nodes recruited up to the given turn.
         ##
        getRecruitedUpToTurn = function(turn)
        {
          return (names(Filter(function(x) x$turn <= turn, private$recruited)))
        },
        
        ##
         # Returns a list with the indices of all models in private$models which are currently valid.
         ##
        getValidModels = function()
        {
          return (unlist(Filter(function(i) private$models[[i]]$isValid(),
                                1:private$settings$nmodels)))
        },

        ##
         # Starts the simulator.
         # Creates the models from new seeds and cold-start them with additional samples.
         ##
        start = function(seeds)
        {
          nseeds <- private$settings$nseeds
          nmodels <- private$settings$nmodels
        
          # Initializes internal data structures.
          private$models <- list()
          private$recruited <- list()
          private$turn <- 1
          private$observed_feats <- NULL
          private$selected_nodes <- NULL
          private$selected_models <- NULL
          private$observed_graph <- ObservedGraph$new(private$settings)
        
          # Creates the models.
          private$display("Creating models...\n")
          private$models <- Map(function(i) Model$new(private$settings, 
                                                      private$settings$models[[i]],
                                                      private$observed_graph,
                                                      private$external_feats),
                                1:nmodels)
          
          # Initializes the action selector.
          private$action_selector <- ActionSelector$new(private$settings, private$models)
          
          # Recruits the seeds.
          private$display("Seeds:\n"); print(seeds)
          for (i in 1:nseeds) {
            private$recruitNode(seeds[i], border_check=FALSE)
            private$turn <- 1 + private$turn
          }
          
          # Runs the cold start.
          i <- 0
          heuristic <- private$settings$initial_heuristic
          private$display(sprintf("Cold-start criterion: %s\n", heuristic))
          while ((i < private$settings$ncold) | (length(private$getValidModels()) == 0)) {
            # Recruits one more node using the configured heuristic.
            node <- private$recruitWithHeuristic()
            private$turn <- 1 + private$turn
            # If "MOD until N zeros", increments i only when zero-labeled nodes are recruited.
            i <- i + (if (heuristic == "modunz") (node$node_info$label == 0) else 1)
            # Shows the status of the cold-start process.
            private$display(sprintf("Cold-start status: %d out of %d\r", i, private$settings$ncold))
          }
          private$display("\n")
        },
        
        ##
         # Recruits one node with the heuristic specified in the settings.
         ##
        recruitWithHeuristic = function()
        {
          heuristic <- private$settings$initial_heuristic
          node_id <- if (heuristic == "random") private$observed_graph$getRandomSuggestion()
                else if (heuristic %in% c("mod", "modunz")) private$observed_graph$getModSuggestion()
          node_info <- private$recruitNode(node_id)
          return (list(node_id=node_id, node_info=node_info))
        },
        
        ##
         #
         ##
        runTurns = function(nturns)
        {
          # Recruits the most prominent node in each turn, until there are no more turns or no more nodes.
          final_turn <- private$turn + nturns
          while (private$turn < final_turn) {
          
            # Reports simulation status.
            if ((!private$settings$output_to_file) |
                (private$settings$output_to_file & ((private$turn %% 10) == 0))) {
              private$display(sprintf("Turn %03d\r", private$turn))
            }
            
            # Performs model selection and recruits the most prominent border node.
            ret <- private$recruitNextNode()
            
            if (ret == private$NEWLY_RECRUITED) {
              # Advances to the next turn.
              private$turn <- 1 + private$turn
            } else if (ret == private$EMPTY_BORDER) {
              # Stops since there are no more nodes to recruit.
              private$display(paste("\nNo more border nodes to recruit (turn ", private$turn, ").\n", sep=""))
              break
            }
          }
          private$display("\n")
        },

        ##
         #
         ##
        recruitNextNode = function()
        {
          # Lists which models are ready.
          valid_models <- private$getValidModels()
          
          # Selects the next node to recruit.
          action <- private$action_selector$selectNextAction(simulation_state = private$getSimulationState(),
                                                             model_ids = valid_models,
                                                             eval_all = private$settings$parallel_updates)
          
          # Checks if the border set is empty.
          if ((length(action) == 1) && is.na(action)) {
            stopifnot(length(private$observed_graph$border) == 0)
            return (private$EMPTY_BORDER)
          }
          
          # Recruits the selected node.
          node_info <- private$recruitNode(action$node_id)
          
          # Registers the selected model.
          private$selected_models <- append(private$selected_models, action$model_id)
          
          # Updates model statistics.
          correct = (node_info$label > 0)
          if (private$settings$parallel_updates) {
            # Updates all models which recommended the selected node.
            Map(function(model_id) private$action_selector$addFeedback(model_id, correct), action$agreers)
          } else {
            # Updates only the selected model.
            private$action_selector$addFeedback(action$model_id, correct)
          }
          
          return (private$NEWLY_RECRUITED)
        },

        ##
         # Wrapper for the Graph::getInfo() method.
         # Returns a list with information about the node with the given ID.
         # TODO: maybe use a cache here?
         ##
        getNodeInfo = function(character_id=NA)
        {
          id <- as.numeric(character_id)
          info <- private$graph$getInfo(id)
          info$neighbors <- as.character(info$neighbors)
          return (info)
        }
    )
)


##
 #
 ##
SimulationState <- R6Class("SimulationState", public = list(
  
  current_turn = NA,
  final_turn = NA,
  observed_graph = NA,
  
  initialize = function(current_turn=NA, final_turn=NA, observed_graph=NA)
  {
    self$current_turn <- current_turn
    self$final_turn <- final_turn
    self$observed_graph <- observed_graph
  }
))
                           

##
 #
 ##
SimulationResult <- R6Class("SimulationResult", public = list(

    total_payoff = NA,
    nrecruited = NA,
    mean_payoff = NA,
    recruited_id = NA,
    observed_feats = NA,
    model_id = NA,

    initialize = function(total_payoff=NA, nrecruited=NA, mean_payoff=NA, recruited_id=NA, model_id=NA, observed_feats=NA)
    {
      self$total_payoff <- total_payoff
      self$nrecruited <- nrecruited
      self$mean_payoff <- mean_payoff
      self$recruited_id <- recruited_id
      self$model_id <- model_id
      self$observed_feats <- observed_feats
    }
))

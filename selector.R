library(R6)

##
 # General action selector class.
 # Encapsulates specific action selector implementations.
 ##
ActionSelector <- R6Class("ActionSelector",
                 
    public = list(
     
        ##
         # Initializes a new action selector instance.
         ##
        initialize = function(settings=NA, models=NA)
        {
          private$models <- models
          private$settings <- settings
          private$selector <- settings$action_selector
          private$selector$init(private$settings)
        },

        ##
         # Given a list of models, returns the ID of the border node selected by the underlying policy.
         # If the border set is empty, returns NA.
         ##
        selectNextAction = function(simulation_state=NA, model_ids=NA, eval_all=FALSE)
        {
          # Selects which models will be considered for action selection.
          selected_models <- private$selector$selectModels(model_ids, simulation_state)
          models_to_eval <- if (eval_all) model_ids else model_ids[selected_models]
          
          # Evaluates the border set using the models.
          evals <- Map(function(model_id) if (!(model_id %in% models_to_eval)) NA
                                          else private$getModelScores(model_id), model_ids)
          
          # Checks if the border set is empty.
          if (any(unlist(Map(function(x) is.null(x), evals)))) {
            return (NA)
          }
          
          # Selects the next action using the policy.
          action <- private$selector$selectAction(evals[selected_models], simulation_state)
          selected_model <- model_ids[selected_models[action$vector_id]]
          selected_node <- action$action_id
          
          # Finds out which models agree on the selected action.
          agreers <- if (eval_all) model_ids[private$selector$getAgreers(selected_node, evals, simulation_state)]
                     else selected_model
          
          return (list(node_id = selected_node,
                       model_id = selected_model,
                       agreers = agreers))
        },
        
        ##
         # Registers feedback for the given model.
         ##
        addFeedback = function(model_id=NA, correct=NA)
        {
          stopifnot(is.logical(correct))
          stopifnot((model_id >= 1) & (model_id <= private$settings$nmodels))
          private$selector$addFeedback(model_id, correct)
        }
    ),
    
    private = list(
      
        settings = NA,              # (Settings) The simulation settings.
        models = NA,                # (list<Model>) A list referencing the same models as that in the Simulator.
        selector = NA,              # A reference to an initialized instance of a supported action selector class.
        
        ##
         # Gets the scores of all border nodes, as estimated by the specified model.
         # Returns NA if the border set is found to be empty.
         ##
        getModelScores = function(model_id=NA)
        {
          rank <- private$models[[model_id]]$evaluateBorder()
          if (is.null(rank)) {return (NULL)}
          return (rank)
        }
    )
)


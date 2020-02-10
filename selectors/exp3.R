library(R6)
source("selectors/base/single_arm.R")

##
 # Exp3 policy for MAB.
 #
 #   P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire. The Nonstochastic
 #   Multi-armed Bandit Problem. SIAM J. Comput. (), 32(1):48–77, 2002.
 ##
PolicyExp3 <- R6Class("PolicyExp3",
                      inherit = SingleArmSelector,
                         
    public = list(
      
        full_w.t_history = NULL,

        ##
         # Creates a new selector instance.
         ##
        initialize = function(gamma=0.1)
        {
          stopifnot((length(gamma) == 1) && is.numeric(gamma))
          stopifnot((gamma >= 0) && (gamma <= 1))
          private$gamma = gamma
        },
      
        ##
         # Initializes the selector instance.
         ##
        init = function(settings=NA)
        {
          private$score_mapper <- settings$score_mapper
          private$full_w.t <- rep(1, settings$nmodels)
          private$horizon <- settings$nattempts
        },
        
        ##
         # Returns the name of this policy.
         ##
        getName = function()
        {
          return (sprintf("Exp3, gamma=%e", private$gamma) )
        },
        
        ##
         # Exp4 uses the scores from all available experts.
         ##
        selectModels = function(model_ids=NA, simulation_state=NA)
        {
          private$w.t <- private$full_w.t[model_ids]

          K = length(model_ids)
          p.t = private$setProbabilities(K, alternate=FALSE)
          idx = sample.int(K,1,prob=p.t)

          private$model_id = model_ids[idx]
          private$p.t = p.t[idx]
          private$K = K

          self$full_w.t_history <- rbind(self$full_w.t_history, private$full_w.t)
          
          return (idx)
        },
        
        
        ##
         # Registers the feedback.
         ##
        addFeedback = function(model_id=NA, correct=NA)
        {
          stopifnot(is.logical(correct))

          # Step 4: compute scaled reward x.hat.t and update weights
          reward <- as.numeric(correct)
          x.hat.t = reward/private$p.t

          private$full_w.t[private$model_id] = private$full_w.t[private$model_id] *
                                               exp(private$gamma * x.hat.t / private$K)

          # #####
          private$clear()
        }
    ),
    
    private = list(
      
        score_mapper = NA,
        gamma = NA,
        full_w.t = NA,

        model_id = NA,
        w.t = NA,
        p.t = NA,
        K = NA,
        horizon = NA,
        step = NA,
        
        ##
         # Unsets private variables which must be reinitialized at each turn.
         ##
        clear = function()
        {
          private$model_id <- NA
          private$w.t <- NA
          private$p.t <- NA
          private$K <- NA
        },

        setProbabilities = function(K, alternate=FALSE) {
          p.t = private$w.t / sum(private$w.t)

          ##
           # Alternate method for setting probabilities in Step 2
           # Brendan McMahan and Matthew Streeter. Tighter bounds for multi-
           # armed bandits with expert advice. In COLT, 2009.
           ##
          if (alternate) {
            delta = 0
            l     = 1
            p.min = private$gamma/K
            inds  = order(p.t)
            for( i in 1:length(inds) ) {
              j = inds[i]
              if (p.t[j] >= p.min) {
                p.t[inds[i:K]] = p.t[inds[i:K]]*(1-delta/l)
                return (p.t)
              } else {
                delta = delta + (p.min - p.t[j])
                l = l - p.t[j]
                p.t[j] = p.min
              }
            }
          } else { # Original method for setting probabilities (Exp3)
            return((1-private$gamma) * p.t + private$gamma/K)
          }
        }
    )
)


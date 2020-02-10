library(R6)

##
 # Exp4 policy for MAB.
 #
 #   P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire. The Nonstochastic
 #   Multi-armed Bandit Problem. SIAM J. Comput. (), 32(1):48–77, 2002.
 ##
PolicyExp4 <- R6Class("PolicyExp4",
                         
    public = list(
      
        full_w.t_history = NULL,

        ##
         # Creates a new selector instance.
         ##
        initialize = function(gamma=0.01)
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
          private$step <- 0
        },
        
        ##
         # Returns the name of this policy.
         ##
        getName = function()
        {
          return (sprintf("Exp4, gamma=%e", private$gamma) )
        },
        
        ##
         # Exp4 uses the scores from all available experts.
         ##
        selectModels = function(model_ids=NA, simulation_state=NA)
        {
          private$model_ids <- model_ids
          private$w.t <- private$full_w.t[model_ids]
          
          self$full_w.t_history <- rbind(self$full_w.t_history, private$full_w.t)
          
          return (1:length(model_ids))
        },
        
        ##
         # Selects an action according to the Exp4 policy.
         ##
        selectAction = function(score_vectors=NA, simulation_state=NA)
        {
          stopifnot(length(score_vectors) == length(private$model_ids))
          
          # Builds the chi.t matrix.
          chi.t <- private$buildChiT(score_vectors, simulation_state)
          K = ncol(chi.t)

          # Step 2:
          #   - integrate probability of each action over all models
          p.t = private$setProbabilities(chi.t, K, TRUE)

          # Step 3: draw action according to p.t
          action_idx <- sample(1:K, 1, prob=p.t)
          action <- colnames(chi.t)[action_idx]

          # save variables needed to scale reward
          private$p.t   <- p.t[action_idx]
          private$chi.t <- chi.t[,action_idx]
          private$K <- K

          # required by Exp4.P
          private$v.t = as.vector(chi.t %*% (1.0/p.t))

          # update internal step counter
          private$step = private$step+1
          
          return (list(vector_id = 1,
                       action_id = action))
        },
        
        ##
         # Registers the feedback.
         ##
        addFeedback = function(model_id=NA, correct=NA)
        {
          stopifnot(is.logical(correct))

          # Step 5: compute scaled reward x.hat.t
          reward <- as.numeric(correct)
          x.hat.t = reward/private$p.t

          # Step 6: compute expected gain y.hat.t and update weights
          y.hat.t = private$chi.t * x.hat.t

          # Exp4
          # private$full_w.t[private$model_ids] = private$w.t * exp(private$gamma * y.hat.t/private$K)

          # Exp4.P (for delta=.05, T=300) # FIXME: get budget
          N = length(y.hat.t)
          #print(private$w.t * exp(private$gamma * y.hat.t/private$K))
          #print(private$w.t *
          #      exp( private$gamma/(2*private$K) * ( y.hat.t + private$v.t*sqrt(log(N/.05)/(private$K*300)) ) )
          #     )

          private$full_w.t[private$model_ids] = private$w.t *
              exp( private$gamma/(2*private$K) *
                   ( y.hat.t + private$v.t*sqrt(log(N/.05)/(private$K*private$horizon)) ) )
          
          # #####
          private$clear()
        }
    ),
    
    private = list(
      
        score_mapper = NA,
        gamma = NA,
        full_w.t = NA,

        model_ids = NA,
        chi.t = NA,
        w.t = NA,
        p.t = NA,
        v.t = NA,
        K = NA,
        horizon = NA,
        step = NA,
        
        ##
         # Unsets private variables which must be reinitialized at each turn.
         ##
        clear = function()
        {
          private$model_ids <- NA
          private$chi.t <- NA
          private$w.t <- NA
          private$p.t <- NA
          private$v.t <- NA
          private$K <- NA
        },

        setProbabilities = function(chi.t, K, alternate=FALSE) {
          # compute normalizing constant W.t
          # W.t = sum(private$w.t)
          p.t = crossprod(chi.t, private$w.t) / sum(private$w.t)
          #matplot(t(chi.t),type='l',lty=1,col=1:5)
          #legend('topright',c('MOD','AS','LR','RF','LN'),lty=1,col=1:5)

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
          } else { # Original method for setting probabilities (Exp4)
            return((1-private$gamma) * p.t + private$gamma/K)
          }
        },
        
        ##
         # Builds the matrix chi.t from a list of score vectors.
         # chi.t[i,j]: score given by expert i to action j in step t.
         ##
        buildChiT = function(score_vectors=NA, simulation_state=NA)
        {
          chi.t <- matrix(NA, nrow=length(score_vectors), ncol=length(score_vectors[[1]]))
          colnames(chi.t) <- names(score_vectors[[1]])
          for (i in 1:length(score_vectors))
            chi.t[i,] <- private$score_mapper$buildDistribution(score_vectors[[i]], simulation_state)
          return (chi.t)
        }
    )
)


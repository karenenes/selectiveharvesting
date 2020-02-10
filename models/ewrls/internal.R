library(R6)

##
 #
 ##
InternalEWRLS <- R6Class("InternalEWRLS",
        public = list(
        	# U is a design matrix where each col represents a observation
        	# d.vec is a vector of responses
        	# beta is the discount factor (weight)
        	# lambda is the regularization constant
			initialize = function(U, d.vec, beta=1, lambda=0) {
        if (is.na(private$beta) || is.na(private$lambda))
          self$setParameters(beta,lambda)
				private$buffer_U <- U
				private$buffer_d <- d.vec
				private$init()
			},
			setParameters = function(beta=1, lambda=0) {
				stopifnot(beta<=1)
				stopifnot(lambda>=0)
				private$beta <- beta
				private$lambda <- lambda
			},
			update = function(u, d) {
				stopifnot(is.vector(u))
				if (!private$trained) {
					private$buffer_U <- cbind(private$buffer_U,u)
					private$buffer_d <- c(private$buffer_d,d)
					private$init()
				} else {
				  beta.inv = 1/private$beta
				  P.u = private$P %*% u

				  r = as.numeric(1+beta.inv*t(u)%*%P.u)
				  k = beta.inv*P.u/r
				  e = as.numeric(d-t(u)%*%private$w)
				  private$w = private$w+k*e
				  private$P = beta.inv*private$P - (k%*%t(k)) * r
				}
			},
			predict = function(test_x) {
				return(crossprod(test_x,private$w))
			},
			coef = function() {
				return(private$w)
			}

        ),
        private = list(
        	P = NULL,
        	w = NULL,
        	trained = FALSE,
        	buffer_U = NULL,
        	buffer_d = NULL,
        	beta = NA,
        	lambda = NA,
			init = function() {
			    i <- ncol(private$buffer_U)
			    B <- diag(private$beta^(i-(1:i)))

			    private$P <- tryCatch(solve(private$buffer_U%*%B%*%t(private$buffer_U) +
			    	       private$beta^i*private$lambda*diag(nrow(private$buffer_U)) ), error=function(e) NULL)

			    if (!is.null(private$P)) {
				    private$w <- private$P %*% private$buffer_U %*% B %*% private$buffer_d
			    	private$buffer_U <- NULL
			    	private$buffer_d <- NULL
			    	private$trained <- TRUE
			    } else {
			    	warning('Fitting failed. Storing intial observations for future fitting attempts.')
			    }
			}
        )
    )

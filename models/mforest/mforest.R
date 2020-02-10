library(R6)
library(rPython)
# load Mondrian Forest module
python.load('models/mforest/mforest_standalone.py')

##
 #
 ##
assign2nparray = function(variable_name,val) {
	if(length(val)>1) {
    # for python.assign to work, arrays must have no names
    dimnames(val) <- c()
    names(val) <- c()

    # assign and convert to np.array
		python.assign(variable_name,val)
		python.exec(sprintf('%s = np.array(%s)',variable_name,variable_name))

    # if val is row/col matrix, use this little trick to turn it into array of arrays
		if(is.matrix(val) && any(dim(val)==1)) {
			python.exec(sprintf('%s = np.reshape(%s,(%d,%d))',variable_name,variable_name,nrow(val),ncol(val)))
		}
	} else {
		# is integer
		if(!grepl("[^[:digit:]]", format(val,  digits = 20, scientific = FALSE))){
			python.exec(sprintf('%s = np.array([%d])',variable_name,val))
		}
		else {
			python.exec(sprintf('%s = np.array([%20e])',variable_name,val))
		}
	}
}

getNparray = function(variable_name) {
	python.exec(sprintf('tmp = %s.tolist()',variable_name))
	return (python.get('tmp'))
}

MForest <- R6Class("MForest",

    public = list(
    
        ##
         # Initializes the model instance.
         ##
        initialize = function(ntree=500)
        {
          private$n_mondrians <- ntree
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
          name <- sprintf("MForest, ntree=%d, lambda=%.2f", private$n_mondrians, private$lambda)
          return (name)
        },
        
        ##
         #
         ##
        fit = function(x, y)
        {
          if (length(y) >= 5 && length(levels(as.factor(y))) > 1) {
            # build template data vector
            N = nrow(x)
            M = ncol(x)
            data <- list(n_dim=M,
              is_sparse=FALSE,
              n_train=N,
              n_class=2
            )
            python.assign('data',data)
            assign2nparray('data["x_train"]',x)
            assign2nparray('data["y_train"]',y)

            # compute lambda such that # decision nodes = log(M)
            private$lambda <- 1.0/(2*M) * (log2(log(M)+1)-1)

            cmd_options <- c('--n_mondrians',as.character(private$n_mondrians),'--n_minibatches',as.character(2))
            python.assign('cmd_options',cmd_options)
            python.exec('settings = process_command_line(cmd_options)')
            python.exec('reset_random_seed(settings)')
            python.exec('param, cache = precompute_minimal(data, settings)')
            python.exec('mf = MondrianForest(settings,data)')

            assign2nparray('train_ids_current_minibatch',1:N-1)
            python.exec('mf.fit(data,train_ids_current_minibatch,settings,param,cache)')

            private$valid = TRUE

            # prepare for predict calls
            assign2nparray('weights_prediction',rep(1.0/private$n_mondrians,private$n_mondrians))
          }

        },
        ##
         #
         ##
        update = function(x, y)
        {
          # prepare for update call
          assign2nparray('tmp_x',as.vector(x))
          assign2nparray('tmp_y',y)

          # FIXME: make this work without appending; seems impossible though
          python.exec('train_ids_current_minibatch = np.array([len(data["y_train"])])')
          python.exec('data["x_train"] = np.vstack((data["x_train"],tmp_x))')
          python.exec('data["y_train"] = np.append(data["y_train"],tmp_y)')

          python.exec('mf.partial_fit(data,train_ids_current_minibatch,settings,param,cache)')
          python.exec('print_forest_stats(mf,settings,data)')
        },
        
        ##
         # 
         ##
        predict = function(x)
        {
          N = nrow(x)
          yhat <- rep(0,N)
          if (private$valid) {
              assign2nparray('test_ids_cumulative',1:N-1)
              assign2nparray('data["x_test"]',x)
              #assign2nparray('data["y_test"]',yhat)

              # FIXME: replace evaluate_predictions by a more objective method 
              python.exec(paste('pred_forest_test = mf.predict(',
                'data, data["x_test"][test_ids_cumulative, :],',
                #'data["y_test"][test_ids_cumulative],',
                    'settings, param, weights_prediction)'))
              yhat = getNparray('pred_forest_test["pred_prob"][:,1]')
          }
          names(yhat) <- rownames(x)
          return(yhat)
        }
    ),

    private = list(
        
        lambda = NA,
        n_mondrians = NA,
        valid = FALSE
    )
)

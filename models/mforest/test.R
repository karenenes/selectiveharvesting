library(rPython)
library(Matrix)
options(error=recover)

assign2nparray = function(variable_name,val) {
	if(length(val)>1) {
		python.assign(variable_name,val)
		python.exec(sprintf('%s = np.array(%s)',variable_name,variable_name))
		if(is.matrix(val) && any(dim(val)==1)) {
			python.exec(sprintf('%s = np.reshape(%s,(%d,%d))',variable_name,variable_name,nrow(val),ncol(val)))
		}
	} else {
		# is integer
		if(!grepl("[^[:digit:]]", format(N,  digits = 20, scientific = FALSE))){
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

# load Mondrian Forest module
python.load('mforest_standalone.py')

# load data to matrix 'features' (features by obs)
z <- load('kickstarter.RData')

ids <- features[1,]
y <- features[2,]
x <- t(features[c(-1,-2),])
sds <- apply(x,MARGIN=2,FUN=sd)
sds[sds==0] <- 1
x <- scale(t(features[c(-1,-2),]), scale=sds)

N <- 1000
x_train <- x[1:N,] # TODO: remove this
y_train <- y[1:N]
x_test <- x[N+(1:N),]
y_test <- y[N+(1:N)]

M <- ncol(x)





# generate data vector
data <- list(n_dim=M,
	is_sparse=FALSE,
	n_train=N,
	n_test=N,
	n_class=2
	)

python.assign('data',data)
assign2nparray('data["x_train"]',x_train)
assign2nparray('data["y_train"]',y_train)
assign2nparray('data["x_test"]',x_test)
assign2nparray('data["y_test"]',y_test)


#assign2nparray('train_ids_current_minibatch',1:first_batch_size-1)
# initialize MForest params
first_batch_size <- 20
lambda <- 1.0/(2*M) * (log2(log(M)+1)-1)
n_mondrians <- 10

# pass arguments to process_command_line
# FIXME: existing bug causes fit to fail if n_minibatches == 1
cmd_options <- c('--n_mondrians',as.character(n_mondrians),'--n_minibatches',as.character(2))
python.assign('cmd_options',cmd_options)
python.exec('settings = process_command_line(cmd_options)')
python.exec('reset_random_seed(settings)')
python.exec('param, cache = precompute_minimal(data, settings)')
python.exec('mf = MondrianForest(settings,data)')

# train
assign2nparray('train_ids_current_minibatch',0:19)
python.exec('mf.fit(data,train_ids_current_minibatch,settings,param,cache)')

assign2nparray('weights_prediction',rep(1.0/n_mondrians,n_mondrians))
assign2nparray('test_ids_cumulative',1:N-1)

weighted_perf <- rep(NA,N)
for(idx in 20:(N-1)) {
	#print(c('==========================>',idx))
	assign2nparray('train_ids_current_minibatch',idx)
	python.exec('mf.partial_fit(data,train_ids_current_minibatch,settings,param,cache)')

	###### prediction

	python.exec(paste('pred_forest_test, metrics_test = mf.evaluate_predictions(',
		'data, data["x_test"][test_ids_cumulative, :],',
		'data["y_test"][test_ids_cumulative],',
   	    'settings, param, weights_prediction)'))
	pred = getNparray('pred_forest_test["pred_prob"][:,1]')
	weighted_perf[idx+1] = crossprod(0.99^(1:N), y_test[order(pred,decreasing=TRUE)])		
}






#train_id=first_batch_size+1
#assign2nparray('train_ids_current_minibatch',train_id-1)
##assign2nparray('data["x_train"]',x[train_id,,drop=FALSE])
##assign2nparray('data["y_train"]',y[train_id])
#python.exec('mf.partial_fit(data,train_ids_current_minibatch,settings,param,cache)')
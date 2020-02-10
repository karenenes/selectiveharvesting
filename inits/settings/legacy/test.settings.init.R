source("inits/problem.init.R")
source("settings.R")

# Available models.
source("models/activesearch/activesearch.R")
source("models/ewrls/ewrls.R")
source("models/listnet/listnet.R")
source("models/logistic/logistic.R")
source("models/mod/mod.R")
source("models/rforest/rforest.p.R")
source("models/rforest/rforest.rf.R")
source("models/svm/svm.R")

# -----------------------------------------------------------------------------------------------
MODELS <- list(MOD$new(),                            # 1
               ActiveSearch$new(),                   # 2
               SVM$new(C='heuristic'),               # 3
               RForest.p$new(ntree=100),             # 4
               ListNet$new(max.iter=100),            # 5
               EWRLS$new(beta=1.0, lambda=1.0),      # 6
               Logistic$new(C='heuristic'))          # 7

# -----------------------------------------------------------------------------------------------
SETTINGS <- list()

# kickstarter
SETTINGS$kickstarter <- Settings$new(problem=PROBLEMS$kickstarter, rng_seed=112358, nattempts=80, models=MODELS)

# dbpedia
SETTINGS$dbpedia <- Settings$new(problem=PROBLEMS$dbpedia, rng_seed=112358, nattempts=80, models=MODELS)

# citeseer
SETTINGS$citeseer <- Settings$new(problem=PROBLEMS$citeseer, rng_seed=112358, nattempts=80, models=MODELS)

# wikipedia
SETTINGS$wikipedia <- Settings$new(problem=PROBLEMS$wikipedia, rng_seed=112358, nattempts=80, models=MODELS)

# donors
SETTINGS$donors <- Settings$new(problem=PROBLEMS$donors, rng_seed=112358, nattempts=80, models=MODELS)


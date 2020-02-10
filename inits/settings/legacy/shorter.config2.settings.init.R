source("inits/problem.init.R")
source("settings.R")

# Available models.
source("models/mod/mod.R")
source("models/activesearch/activesearch.R")
source("models/svm/svm.R")
source("models/rforest/rforest.p.R")
source("models/rforest/rforest.rf.R")
source("models/listnet/listnet.R")

#################################################################################################
RFOREST.P  <- RForest.p$new(ntree=100)
RFOREST.RF <- RForest.rf$new(ntree=100)

MODELS <- list(MOD$new(),
               ActiveSearch$new(),
               SVM$new(C='heuristic'),
               NA,
               ListNet$new(max.iter=100))

MODELS.P  <- Map(function(m) if (is.logical(m) && is.na(m)) RFOREST.P  else m, MODELS)
MODELS.RF <- Map(function(m) if (is.logical(m) && is.na(m)) RFOREST.RF else m, MODELS)
#################################################################################################

SETTINGS <- list()

# ===============================================================================================
# flickr
SETTINGS$flickr <- Settings$new(problem=PROBLEMS$flickr, rng_seed=112358, nattempts=1400, models=MODELS.P)
# youtube
SETTINGS$youtube <- Settings$new(problem=PROBLEMS$youtube, rng_seed=112358, nattempts=1400, models=MODELS.P)
# blogcatalog
SETTINGS$blogcatalog <- Settings$new(problem=PROBLEMS$blogcatalog, rng_seed=112358, nattempts=1400, models=MODELS.P)
# ===============================================================================================
# dbpedia
SETTINGS$dbpedia <- Settings$new(problem=PROBLEMS$dbpedia, rng_seed=112358, nattempts=700, models=MODELS.P)
# citeseer
SETTINGS$citeseer <- Settings$new(problem=PROBLEMS$citeseer, rng_seed=112358, nattempts=1500, models=MODELS.P)
# wikipedia
SETTINGS$wikipedia <- Settings$new(problem=PROBLEMS$wikipedia, rng_seed=112358, nattempts=400, models=MODELS.P)
# ===============================================================================================
# donors
SETTINGS$donors <- Settings$new(problem=PROBLEMS$donors, rng_seed=112358, nattempts=150, models=MODELS.P)
# kickstarter
SETTINGS$kickstarter <- Settings$new(problem=PROBLEMS$kickstarter, rng_seed=112358, nattempts=700, models=MODELS.P)
# ===============================================================================================
# dblp
SETTINGS$dblp <- Settings$new(problem=PROBLEMS$dblp, rng_seed=112358, nattempts=1200, models=MODELS.RF)
# lj
SETTINGS$lj <- Settings$new(problem=PROBLEMS$lj, rng_seed=112358, nattempts=1200, models=MODELS.RF)
# ===============================================================================================

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

#################################################################################################
RFOREST.P  <- RForest.p$new(ntree=100)
RFOREST.RF <- RForest.rf$new(ntree=100)
               
MODELS <- list(MOD$new(),                            # 1
               ActiveSearch$new(),                   # 2
               SVM$new(C='heuristic'),               # 3
               NA,                                   # 4
               ListNet$new(max.iter=100),            # 5
               EWRLS$new(beta=0.9, lambda=1.0),      # 6
               EWRLS$new(beta=1.0, lambda=1.0),      # 7
               Logistic$new(C='heuristic'))          # 8

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
SETTINGS$kickstarter <- Settings$new(problem=PROBLEMS$kickstarter, rng_seed=112358, nattempts=700, models=MODELS.RF)
# ===============================================================================================
# dblp
SETTINGS$dblp <- Settings$new(problem=PROBLEMS$dblp, rng_seed=112358, nattempts=1200, models=MODELS.RF)
# lj
SETTINGS$lj <- Settings$new(problem=PROBLEMS$lj, rng_seed=112358, nattempts=1200, models=MODELS.RF)
# ===============================================================================================
# ===============================================================================================
# TANE
# ===============================================================================================
# blogcatalog
SETTINGS$blogcatalog <- Settings$new(problem=PROBLEMS$blogcatalog, rng_seed=112358, nattempts=1400, models=MODELS.P)
# flickr
SETTINGS$flickr <- Settings$new(problem=PROBLEMS$flickr, rng_seed=112358, nattempts=4000, models=MODELS.P)
# ppi
SETTINGS$ppi <- Settings$new(problem=PROBLEMS$ppi, rng_seed=112358, nattempts=2000, models=MODELS.P)
# cora
SETTINGS$cora <- Settings$new(problem=PROBLEMS$cora, rng_seed=112358, nattempts=1100, models=MODELS.P)
# amazon
SETTINGS$amazon <- Settings$new(problem=PROBLEMS$amazon, rng_seed=112358, nattempts=2000, models=MODELS.P)
# ===============================================================================================


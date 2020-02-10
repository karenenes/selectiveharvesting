source("inits/problem.init.R")
source("settings.R")

# Available models.
source("models/rforest/rforest.p.R")
source("models/rforest/rforest.rf.R")

SETTINGS <- list()

# -----------------------------------------------------------------------------------------------
# flickr
SETTINGS$flickr <- Settings$new(problem=PROBLEMS$flickr, rng_seed=112358, nattempts=1400,
                                models=list(RForest.p$new(ntree=500)))

# youtube
SETTINGS$youtube <- Settings$new(problem=PROBLEMS$youtube, rng_seed=112358, nattempts=1400,
                                 models=list(RForest.p$new(ntree=500)))

# blogcatalog
SETTINGS$blogcatalog <- Settings$new(problem=PROBLEMS$blogcatalog, rng_seed=112358, nattempts=1400,
                                     models=list(RForest.p$new(ntree=500)))

# -----------------------------------------------------------------------------------------------
# dbpedia
SETTINGS$dbpedia <- Settings$new(problem=PROBLEMS$dbpedia, rng_seed=112358, nattempts=1000,
                                 models=list(RForest.p$new(ntree=500)))

# citeseer
SETTINGS$citeseer <- Settings$new(problem=PROBLEMS$citeseer, rng_seed=112358, nattempts=1400,
                                  models=list(RForest.p$new(ntree=500)))

# wikipedia
SETTINGS$wikipedia <- Settings$new(problem=PROBLEMS$wikipedia, rng_seed=112358, nattempts=1000,
                                   models=list(RForest.p$new(ntree=500)))

# -----------------------------------------------------------------------------------------------

# donors
SETTINGS$donors <- Settings$new(problem=PROBLEMS$donors, rng_seed=112358, nattempts=300,
                                models=list(RForest.p$new(ntree=500)))

# donors amounts
SETTINGS$donors_amounts <- Settings$new(problem=PROBLEMS$donors_amounts, rng_seed=112358, nattempts=300,
                                        models=list(RForest.p$new(ntree=500)))

# kickstarter
SETTINGS$kickstarter <- Settings$new(problem=PROBLEMS$kickstarter, rng_seed=112358, nattempts=1200,
                                     models=list(RForest.p$new(ntree=500)))

# amazon
SETTINGS$amazon <- Settings$new(problem=PROBLEMS$amazon, rng_seed=112358, nattempts=400,
                                models=list(RForest.p$new(ntree=500)))

# dblp
SETTINGS$dblp <- Settings$new(problem=PROBLEMS$dblp, rng_seed=112358, nattempts=1200,
                              models=list(RForest.rf$new(ntree=100)))

# lj
SETTINGS$lj <- Settings$new(problem=PROBLEMS$lj, rng_seed=112358, nattempts=1200,
                            models=list(RForest.rf$new(ntree=100)))


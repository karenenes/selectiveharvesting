source("inits/problem.init.R")
source("settings.R")

# Available models.
source("models/listnet/listnet.R")

#################################################################################################
MODELS <- list(ListNet$new(max.iter=100))
#################################################################################################

SETTINGS <- list()

# ===============================================================================================
# flickr
SETTINGS$flickr <- Settings$new(problem=PROBLEMS$flickr, rng_seed=112358, nattempts=1400, models=MODELS)
# youtube
SETTINGS$youtube <- Settings$new(problem=PROBLEMS$youtube, rng_seed=112358, nattempts=1400, models=MODELS)
# blogcatalog
SETTINGS$blogcatalog <- Settings$new(problem=PROBLEMS$blogcatalog, rng_seed=112358, nattempts=1400, models=MODELS)
# ===============================================================================================
# dbpedia
SETTINGS$dbpedia <- Settings$new(problem=PROBLEMS$dbpedia, rng_seed=112358, nattempts=700, models=MODELS)
SETTINGS$dbpedia2 <- Settings$new(problem=PROBLEMS$dbpedia2, rng_seed=112358, nattempts=700, models=MODELS)
SETTINGS$dbpedia3 <- Settings$new(problem=PROBLEMS$dbpedia3, rng_seed=112358, nattempts=700, models=MODELS)
# citeseer
SETTINGS$citeseer <- Settings$new(problem=PROBLEMS$citeseer, rng_seed=112358, nattempts=1500, models=MODELS)
SETTINGS$citeseer2 <- Settings$new(problem=PROBLEMS$citeseer2, rng_seed=112358, nattempts=1500, models=MODELS)
SETTINGS$citeseer3 <- Settings$new(problem=PROBLEMS$citeseer3, rng_seed=112358, nattempts=1500, models=MODELS)
# wikipedia
SETTINGS$wikipedia <- Settings$new(problem=PROBLEMS$wikipedia, rng_seed=112358, nattempts=400, models=MODELS)
SETTINGS$wikipedia2 <- Settings$new(problem=PROBLEMS$wikipedia2, rng_seed=112358, nattempts=400, models=MODELS)
SETTINGS$wikipedia3 <- Settings$new(problem=PROBLEMS$wikipedia3, rng_seed=112358, nattempts=400, models=MODELS)
# ===============================================================================================
# donors
SETTINGS$donors <- Settings$new(problem=PROBLEMS$donors, rng_seed=112358, nattempts=150, models=MODELS)
SETTINGS$donors2 <- Settings$new(problem=PROBLEMS$donors2, rng_seed=112358, nattempts=150, models=MODELS)
SETTINGS$donors3 <- Settings$new(problem=PROBLEMS$donors3, rng_seed=112358, nattempts=150, models=MODELS)
# kickstarter
SETTINGS$kickstarter <- Settings$new(problem=PROBLEMS$kickstarter, rng_seed=112358, nattempts=700, models=MODELS)
SETTINGS$kickstarter2 <- Settings$new(problem=PROBLEMS$kickstarter2, rng_seed=112358, nattempts=700, models=MODELS)
SETTINGS$kickstarter3 <- Settings$new(problem=PROBLEMS$kickstarter3, rng_seed=112358, nattempts=700, models=MODELS)
# ===============================================================================================
# dblp
SETTINGS$dblp <- Settings$new(problem=PROBLEMS$dblp, rng_seed=112358, nattempts=1200, models=MODELS)
# lj
SETTINGS$lj <- Settings$new(problem=PROBLEMS$lj, rng_seed=112358, nattempts=1200, models=MODELS)
# ===============================================================================================

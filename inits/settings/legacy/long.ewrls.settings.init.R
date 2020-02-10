source("inits/problem.init.R")
source("settings.R")

# Available models.
source("models/ewrls/ewrls.R")

SETTINGS <- list()

# -----------------------------------------------------------------------------------------------
# flickr
SETTINGS$flickr <- Settings$new(problem=PROBLEMS$flickr, rng_seed=112358, nattempts=1400,
                                models=list(EWRLS$new(beta=0.99, lambda=1.0)))

# youtube
SETTINGS$youtube <- Settings$new(problem=PROBLEMS$youtube, rng_seed=112358, nattempts=1400,
                                 models=list(EWRLS$new(beta=0.99, lambda=1.0)))

# blogcatalog
SETTINGS$blogcatalog <- Settings$new(problem=PROBLEMS$blogcatalog, rng_seed=112358, nattempts=1400,
                                     models=list(EWRLS$new(beta=0.99, lambda=1.0)))

# -----------------------------------------------------------------------------------------------
# dbpedia
SETTINGS$dbpedia <- Settings$new(problem=PROBLEMS$dbpedia, rng_seed=112358, nattempts=1000,
                                 models=list(EWRLS$new(beta=0.99, lambda=1.0)))

# citeseer
SETTINGS$citeseer <- Settings$new(problem=PROBLEMS$citeseer, rng_seed=112358, nattempts=1500,
                                  models=list(EWRLS$new(beta=0.99, lambda=1.0)))

# wikipedia
SETTINGS$wikipedia <- Settings$new(problem=PROBLEMS$wikipedia, rng_seed=112358, nattempts=1000,
                                   models=list(EWRLS$new(beta=0.99, lambda=1.0)))

# -----------------------------------------------------------------------------------------------
# donors
SETTINGS$donors <- Settings$new(problem=PROBLEMS$donors, rng_seed=112358, nattempts=300,
                                models=list(EWRLS$new(beta=0.99, lambda=1.0)))

# donors amounts
SETTINGS$donors_amounts <- Settings$new(problem=PROBLEMS$donors_amounts, rng_seed=112358, nattempts=300,
                                        models=list(EWRLS$new(beta=0.99, lambda=1.0)))

# kickstarter
SETTINGS$kickstarter <- Settings$new(problem=PROBLEMS$kickstarter, rng_seed=112358, nattempts=1500,
                                     models=list(EWRLS$new(beta=0.99, lambda=1.0)))

# amazon
SETTINGS$amazon <- Settings$new(problem=PROBLEMS$amazon, rng_seed=112358, nattempts=400,
                                models=list(EWRLS$new(beta=0.99, lambda=1.0)))

# dblp
SETTINGS$dblp <- Settings$new(problem=PROBLEMS$dblp, rng_seed=112358, nattempts=1250,
                              models=list(EWRLS$new(beta=0.99, lambda=1.0)))

# lj
SETTINGS$lj <- Settings$new(problem=PROBLEMS$lj, rng_seed=112358, nattempts=1250,
                            models=list(EWRLS$new(beta=0.99, lambda=1.0)))


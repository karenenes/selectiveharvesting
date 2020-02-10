source("problem.init.R")
source("settings.R")

# Available models.
source("models/logistic/logistic.R")

SETTINGS <- list()

initMAB = function() {
  return (list(
Logistic$new(C="heuristic")
    )
  )
}

# -----------------------------------------------------------------------------------------------
# flickr
SETTINGS$flickr <- Settings$new(problem=PROBLEMS$flickr, rng_seed=112358, nattempts=1400,
                                models=initMAB())

# youtube
SETTINGS$youtube <- Settings$new(problem=PROBLEMS$youtube, rng_seed=112358, nattempts=1400,
                                models=initMAB())

# blogcatalog
SETTINGS$blogcatalog <- Settings$new(problem=PROBLEMS$blogcatalog, rng_seed=112358, nattempts=1400,
                                models=initMAB())

# -----------------------------------------------------------------------------------------------
# dbpedia
SETTINGS$dbpedia <- Settings$new(problem=PROBLEMS$dbpedia, rng_seed=112358, nattempts=1000,
                                models=initMAB())

# citeseer
SETTINGS$citeseer <- Settings$new(problem=PROBLEMS$citeseer, rng_seed=112358, nattempts=1400,
                                models=initMAB())

# wikipedia
SETTINGS$wikipedia <- Settings$new(problem=PROBLEMS$wikipedia, rng_seed=112358, nattempts=1000,
                                models=initMAB())

# -----------------------------------------------------------------------------------------------
# donors
SETTINGS$donors <- Settings$new(problem=PROBLEMS$donors, rng_seed=112358, nattempts=300,
                                models=initMAB())

# donors amounts
SETTINGS$donors_amounts <- Settings$new(problem=PROBLEMS$donors_amounts, rng_seed=112358, nattempts=300,
                                models=initMAB())

# kickstarter
SETTINGS$kickstarter <- Settings$new(problem=PROBLEMS$kickstarter, rng_seed=112358, nattempts=1200,
                                models=initMAB())

# amazon
SETTINGS$amazon <- Settings$new(problem=PROBLEMS$amazon, rng_seed=112358, nattempts=400,
                                models=initMAB())

# Supported score mappers.
source("score_mappers/geom_abs.R")
source("score_mappers/geom_dyn.R")
source("score_mappers/geom_rel.R")
source("score_mappers/max_score.R")
#source("score_mappers/max_explore.R")
source("score_mappers/pl_abs.R")
source("score_mappers/pl_dyn.R")
source("score_mappers/pl_rel.R")
source("score_mappers/pl_static.R")


SCORE_MAPPERS <- list()

SCORE_MAPPERS$max       <- ScoreMapperMaxScore$new()
#SCORE_MAPPERS$max_exp   <- ScoreMapperMaxExplore$new()

SCORE_MAPPERS$geom_abs   <- ScoreMapperGeometricAbsolute$new()
SCORE_MAPPERS$geom_dyn   <- ScoreMapperGeometricDynamic$new()
SCORE_MAPPERS$geom_rel   <- ScoreMapperGeometricRelative$new()

SCORE_MAPPERS$pl_abs     <- ScoreMapperPowerLawAbsolute$new()
SCORE_MAPPERS$pl_dyn     <- ScoreMapperPowerLawDynamic$new()
SCORE_MAPPERS$pl_rel     <- ScoreMapperPowerLawRelative$new()
SCORE_MAPPERS$pl_static  <- ScoreMapperPowerLawStatic$new()

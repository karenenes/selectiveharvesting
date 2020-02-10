# -- KAREN
source("features.R")

FEATURE_SETS <- list()

# embedding features.
# FEATURE_SETS$all_3 <- Pay2RecruitFeatures$new(include_structural=TRUE, include_triangles=TRUE, include_most=TRUE, 
#                                               include_rw=TRUE, include_counts=TRUE, include_fractions=TRUE, 
#                                               include_attrib_fractions=TRUE, include_embedding=TRUE)
# 
# FEATURE_SETS$only_embedding <- Pay2RecruitFeatures$new(include_structural=FALSE, include_triangles=FALSE, include_most=FALSE, 
#                                               include_rw=FALSE, include_counts=FALSE, include_fractions=FALSE, 
#                                               include_attrib_fractions=FALSE, include_embedding=TRUE)
# 
# FEATURE_SETS$embedding_attrib_features <- Pay2RecruitFeatures$new(include_structural=FALSE, include_triangles=FALSE, include_most=FALSE, 
#                                               include_rw=FALSE, include_counts=FALSE, include_fractions=FALSE, 
#                                               include_attrib_fractions=TRUE, include_embedding=TRUE)
# 
# FEATURE_SETS$embedding_structural <- Pay2RecruitFeatures$new(include_structural=TRUE, include_triangles=TRUE, include_most=TRUE, 
#                                               include_rw=TRUE, include_counts=TRUE, include_fractions=TRUE, 
#                                               include_attrib_fractions=FALSE, include_embedding=TRUE)



# All features.
FEATURE_SETS$all <- Pay2RecruitFeatures$new(include_structural=TRUE, include_triangles=TRUE, include_most=TRUE, include_rw=TRUE,
                                            include_counts=TRUE, include_fractions=TRUE, include_attrib_fractions=TRUE)#, include_embedding=FALSE)

# All features but one of the structural ones.
FEATURE_SETS$no_triangles <-  Pay2RecruitFeatures$new(include_structural=TRUE, include_triangles=FALSE, include_most=TRUE, include_rw=TRUE,
                                                      include_counts=TRUE, include_fractions=TRUE, include_attrib_fractions=TRUE)#, include_embedding=FALSE)
FEATURE_SETS$no_most <-       Pay2RecruitFeatures$new(include_structural=TRUE, include_triangles=TRUE, include_most=FALSE, include_rw=TRUE,
                                                      include_counts=TRUE, include_fractions=TRUE, include_attrib_fractions=TRUE)#, include_embedding=FALSE)
FEATURE_SETS$no_rw <-         Pay2RecruitFeatures$new(include_structural=TRUE, include_triangles=TRUE, include_most=TRUE, include_rw=FALSE,
                                                      include_counts=TRUE, include_fractions=TRUE, include_attrib_fractions=TRUE)#, include_embedding=FALSE)
FEATURE_SETS$no_counts <-     Pay2RecruitFeatures$new(include_structural=TRUE, include_triangles=TRUE, include_most=TRUE, include_rw=TRUE,
                                                      include_counts=FALSE, include_fractions=TRUE, include_attrib_fractions=TRUE)#, include_embedding=FALSE)
FEATURE_SETS$no_fractions <-  Pay2RecruitFeatures$new(include_structural=TRUE, include_triangles=TRUE, include_most=TRUE, include_rw=TRUE,
                                                      include_counts=TRUE, include_fractions=FALSE, include_attrib_fractions=TRUE)#, include_embedding=FALSE)

# Only (all of the) structural features (same as "no_attrib_fractions").
FEATURE_SETS$only_structural <- Pay2RecruitFeatures$new(include_structural=TRUE, include_triangles=TRUE, include_most=TRUE, include_rw=TRUE,
                                                        include_counts=TRUE, include_fractions=TRUE, include_attrib_fractions=FALSE)#, include_embedding=FALSE)

# Only attribute features (same as "no_structural").
FEATURE_SETS$only_attrib_features <- Pay2RecruitFeatures$new(include_structural=FALSE, include_triangles=FALSE, include_most=FALSE, include_rw=FALSE,
                                                             include_counts=FALSE, include_fractions=FALSE, include_attrib_fractions=TRUE)#, include_embedding=FALSE)

# Only each single structural feature.
FEATURE_SETS$only_triangles <-  Pay2RecruitFeatures$new(include_structural=TRUE, include_triangles=TRUE, include_most=FALSE, include_rw=FALSE,
                                                        include_counts=FALSE, include_fractions=FALSE, include_attrib_fractions=FALSE)#, include_embedding=FALSE)
FEATURE_SETS$only_most <-       Pay2RecruitFeatures$new(include_structural=TRUE, include_triangles=FALSE, include_most=TRUE, include_rw=FALSE,
                                                        include_counts=FALSE, include_fractions=FALSE, include_attrib_fractions=FALSE)#, include_embedding=FALSE)
FEATURE_SETS$only_rw <-         Pay2RecruitFeatures$new(include_structural=TRUE, include_triangles=FALSE, include_most=FALSE, include_rw=TRUE,
                                                        include_counts=FALSE, include_fractions=FALSE, include_attrib_fractions=FALSE)#, include_embedding=FALSE)
FEATURE_SETS$only_counts <-     Pay2RecruitFeatures$new(include_structural=TRUE, include_triangles=FALSE, include_most=FALSE, include_rw=FALSE,
                                                        include_counts=TRUE, include_fractions=FALSE, include_attrib_fractions=FALSE)#, include_embedding=FALSE)
FEATURE_SETS$only_fractions <-  Pay2RecruitFeatures$new(include_structural=TRUE, include_triangles=FALSE, include_most=FALSE, include_rw=FALSE,
                                                        include_counts=FALSE, include_fractions=TRUE, include_attrib_fractions=FALSE)#, include_embedding=FALSE)

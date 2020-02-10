# -- karen
library(R6)

##
 # Selects which features from the Pay2Recruit library to use.
 ##
Pay2RecruitFeatures <- R6Class("Pay2RecruitFeatures", 
                               public = list(
    include_structural = NA,
    include_triangles = NA,
    include_most = NA,
    include_rw = NA,
    include_counts = NA,
    include_fractions = NA,
    include_attrib_fractions = NA,
    #include_embedding = NA, 
    
    ##
     # Initializes a new instance.
     ##
    initialize = function(include_structural=NA, include_triangles=NA, include_most=NA, include_rw=NA,
                          include_counts=NA, include_fractions=NA, include_attrib_fractions=NA)#, include_embedding=NA)
    {
      # Asserts logical values (TRUE or FALSE) have been assigned to all parameters.
      arg_values <- unlist(Map(function(arg) eval(parse(text=arg)), names(formals())))
      stopifnot(all(unlist(Map(function(arg) !is.na(arg) && is.logical(arg), arg_values))))
      
      self$include_structural <- include_structural
      self$include_triangles <- include_triangles
      self$include_most <- include_most
      self$include_rw <- include_rw
      self$include_counts <- include_counts
      self$include_fractions <- include_fractions
      self$include_attrib_fractions <- include_attrib_fractions
      #self$include_embedding <- include_embedding
    }
  )
)

library(R6)

##
 #
 ##
Problem <- R6Class("Problem",

    public = list(
    
        name = NA,
        graph_file = NA,
        attrib_file = NA,
        accept_trait = NA,  # Line (starting at 0) in the atrributes file containing the target nodes community.
        target_size = NA,
        nattribs = NA,
        amount_file = NA,
        
        ##
         # TODO: target_size can be removed. One can implement Graph::getTargetSize() and get its value after
         # the full graph has been loaded. Also, can natrribs be removed, too?
         ##
        initialize = function(name, graph_file, attrib_file, accept_trait, target_size, nattribs, amount_file="")
        {
          self$name <- name
          self$graph_file <- graph_file
          self$attrib_file <- attrib_file
          self$amount_file <- amount_file
          self$accept_trait <- accept_trait
          self$target_size <- target_size
          self$nattribs <- nattribs
        }
        
       ##
        # TODO: validator method.
        ##
    )
)


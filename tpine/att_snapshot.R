#-- karen
#tpine-locality - tpine
library(tidyverse)
library(stringr)

#load_att <- function(problemName, nodes_to_charge){
    
      problemName = "donors"
      nodes_to_charge <- c(5, 7, 11) 
      
      read_att <- readLines(gzfile("../../../data/datasets/gccs/donors/donors.att.txt.gz", open = "r"))
      read_att <-  str_split(read_att[1:5],"\t")
      read_att_list <- c()
      read_att_list_match <- list()
      
      aux_empty <- as.matrix(c(""))
      
      for(i in 1:length(read_att)){
        read_att_list[[i]] <- unlist(str_split(read_att[[i]],pattern = " "))
        read_att_list_match[[i]] <- matrix("")
        
        for (j in 1:length(nodes_to_charge)) {
          aux_att <- c()
          aux_att <- str_extract_all(read_att_list[[i]], pattern = paste("^", nodes_to_charge[j], "$", sep = ""), simplify = TRUE)
          
          if(!is_empty(aux_att)){
            read_att_list_match[[i]] <- rbind(read_att_list_match[[i]], aux_att)  
          }
        }
        
        read_att_list_match[[i]] <- t(read_att_list_match[[i]])
      }
      
      
      
      write.table(read_att_list_match, file = paste("tpine/att_list/", problemName,".attlist", sep = ""), row.names = F, col.names = F)
      
      
      #att_nodes = paste("tane/emb/", problemName, ".att.tpine.txt.gz", sep = "")
      
      #return(att_nodes)
#}
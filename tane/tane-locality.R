#-- karen

library("Matrix")

source("tane/problems.R")
source("tane/problems_turn.R")

get_tane_locality <- function(nodes_to_charge, problemName, algorithm_type){
   
   Nturn_tane <- Nturn_tane + 1
  
   load_adj_matrix <- load_problem(problemName)
   TotalTurn <- load_TotalTurn(problemName, algorithm_type)
   
   edge_in <- c()
   
   for(k in 1:length(nodes_to_charge)){
     idx_i = load_adj_matrix@p[nodes_to_charge[k]]+1
     idx_f = as.numeric(load_adj_matrix@p[nodes_to_charge[k]+1])
     
     idx = as.vector(load_adj_matrix@i[idx_i:idx_f])
     
     edge_in <- rbind(edge_in, cbind(rep.int(nodes_to_charge[k], times = length(idx)), idx+1))
     
   }

   edge_out <- edge_in[, (2:1)]
   edge_list <- rbind(edge_in, edge_out)
 
   ##saves edge_list as input for tane
   write.table(edge_list, file = paste("tane/graph/", problemName,".edgelist", sep = ""), row.names = F, col.names = F)
  
   #runs tane
   sink(paste("tane/emb/", problemName, ".emb", sep = ""))
   sink()
   
   #browser()
   if(Nturn_tane == 1){
      #roda tane pro turn mais recente
      #browser()
      system(paste("tane/node2vec -silent:T -dy -ct:", Nturn," -tt:", TotalTurn, " -i:tane/graph/", problemName, ".edgelist -o:tane/emb/", problemName, ".emb", sep = ""), wait = T)
      
   }else{
      #roda tane pro turn mais recente
      system(paste("tane/node2vec -silent:T -dy -ct:", Nturn," -tt:", TotalTurn, " -pwa:tane/emb/", problemName,"emb.walk -pwe:tane/emb/", problemName, ".emb -i:tane/graph/", problemName, ".edgelist -o:tane/emb/", problemName, ".emb", sep = ""), wait = T)
   }
   
   append_x = read.table(paste("tane/emb/", problemName, ".emb", sep = ""), skip = 1, header = F, sep = " ")
   append_x = as.matrix(append_x)
   
   return(append_x)
  }
  
  
  
  
  

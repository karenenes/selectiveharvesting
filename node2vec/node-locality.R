#-- karen

library("Matrix")

source("node2vec/problems.R")

get_node_locality <- function(nodes_to_charge, problemName){
  
   load_adj_matrix <- load_problem(problemName)
   
   edge_in <- c()
   
   for(k in 1:length(nodes_to_charge)){
     idx_i = load_adj_matrix@p[nodes_to_charge[k]]+1
     idx_f = as.numeric(load_adj_matrix@p[nodes_to_charge[k]+1])
     
     idx = as.vector(load_adj_matrix@i[idx_i:idx_f])
     
     edge_in <- rbind(edge_in, cbind(rep.int(nodes_to_charge[k], times = length(idx)), idx+1))
     
   }

   edge_out <- edge_in[, (2:1)]
   edge_list <- rbind(edge_in, edge_out)
 
   ##saves edge_list as input for node2vec
   ##salva um por um por turn -- usado para criar os snapshots e para testar o n2v a cada rodada
   ##write.table(edge_list, file = paste("node2vec/graph/", problemName,"_", Nturn,".edgelist", sep = ""), row.names = F, col.names = F)
   ## -- comentar daqui pra baixo para salvar snapshots -- ##
   #salva o turn mais recente
   write.table(edge_list, file = paste("node2vec/graph/", problemName,".edgelist", sep = ""), row.names = F, col.names = F)
  
   #runs node2vec
   sink(paste("node2vec/emb/", problemName, ".emb", sep = ""))
   sink()
   
   ##python implementation
   #system(paste("python -W ignore node2vec/src/main.py --input node2vec/graph/", problemName, ".edgelist --output node2vec/emb/", problemName, ".emb --dimensions 2", sep = ""), wait = T)
   
   #c++ implementation
   #roda n2v pra turn atual
   #system(paste("node2vec/node2vec -silent:T -i:node2vec/graph/", problemName,"_", Nturn, ".edgelist -o:node2vec/emb/", problemName, ".emb -d:2 -p:0.5", sep = ""), wait = T)
   #roda n2v pro turn mais recente
   
   
   system(paste("node2vec/node2vec -silent:T -i:node2vec/graph/", problemName, ".edgelist -o:node2vec/emb/", problemName, ".emb -d:2 -p:0.5", sep = ""), wait = T)
   
   append_x = read.table(paste("node2vec/emb/", problemName, ".emb", sep = ""), skip = 1, header = F, sep = " ")
   append_x = as.matrix(append_x)
   
   return(append_x)

  }
  
  
  
  
  

#-- karen
library("Matrix")

get_oracle <- function(nodes_to_charge, problemName){
  
  #nodes_to_charge <- c(49, 35, 427)
  #problemName <- "donors"
  
  #browser()
  get_emb = NA
  get_emb = read.table(paste("oracle/emb/", problemName, "128p1.emb", sep = ""), skip = 1, header = F, sep = " ")
  get_emb = as.matrix(get_emb)
  
  idx_emb = match(nodes_to_charge, get_emb[,1])
  
  select_emb <- get_emb[idx_emb, -c(1)]
  if(!(is.matrix(select_emb))){
    select_emb <- t(as.matrix(select_emb))
  }
  return(select_emb)
  
}


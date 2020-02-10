#-- karen
#get-oracle

load_problem_emb <- function(problemName){
  
  if(problemName == "citeseer"){
    load_matrix <- load("emb/citeseer128p1.emb")
  }
  if(problemName == "donors"){
    load_matrix <- load("emb/donors128p1.emb")
  }
  if(problemName == "dbpedia"){
    load_matrix <- load("emb/dbpedia128p1.emb")
  }
  if(problemName == "kickstarter"){
    load_matrix <- load("emb/kickstarter128p1.emb")
  }
  if(problemName == "wikipedia"){
    load_matrix <- load("emb/wikipedia128p1.emb")
  }
  #if(problemName == "test"){
  #  load_matrix <- load("emb/karate.emb")
  #}
  
  return(adj_matrix)
}
#-- karen
#node-locality - n2v

load_problem <- function(problemName){
  
  if(problemName == "citeseer"){
    load_matrix <- load("../../../data/datasets/gccs/citeseer/labeled/citeseer_matrix.Rdata")
  }
  if(problemName == "donors"){
    load_matrix <- load("../../../data/datasets/gccs/donors/donors_matrix.Rdata")
  }
  if(problemName == "dbpedia"){
    load_matrix <- load("../../../data/datasets/wang2013/dbpedia/done/dbpedia_matrix.Rdata")
  }
  if(problemName == "kickstarter"){
    load_matrix <- load("../../../data/datasets/kickstarter/kickstarter_matrix.Rdata")
  }
  if(problemName == "wikipedia"){
    load_matrix <- load("../../../data/datasets/wang2013/wikipedia/done/wikipedia_matrix.Rdata")
  }
  if(problemName == "test"){
    load_matrix <- load("../../../data/datasets/test/test_matrix.Rdata")
  }
  if(problemName == "blogcatalog"){
    load_matrix <- load("../../../data/datasets/new/blogcatalog/blogcatalog_matrix.Rdata")
  }
  if(problemName == "flickr"){
    load_matrix <- load("../../../data/datasets/new/flickr/flickr_matrix.Rdata")
  }
  if(problemName == "ppi"){
    load_matrix <- load("../../../data/datasets/new/ppi/ppi_matrix.Rdata")
  }
  if(problemName == "cora"){
    load_matrix <- load("../../../data/datasets/new/cora/cora_matrix.Rdata")
  }
  if(problemName == "lj"){
    load_matrix <- load("../../../data/datasets/lj/lj_matrix.Rdata")
  }
  if(problemName == "dblp"){
    load_matrix <- load("../../../data/datasets/dblp/dblp_matrix.Rdata")
  }
  if(problemName == "ppi"){
    load_matrix <- load("../../../data/datasets/new/ppi/ppi_matrix.Rdata")
  }
  if(problemName == "amazon"){
    load_matrix <- load("../../../data/datasets/new/amazon/amazon_matrix.Rdata")
  }
  return(adj_matrix)
}
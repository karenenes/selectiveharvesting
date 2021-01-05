#-- karen
#node-locality - n2v

load_TotalTurn <- function(problemName, algorithm_type){
  
  if(problemName == "citeseer"){
    turns <- 1500
  }
  if(problemName == "donors"){
    if(algorithm_type == 'ComplModel'){
      turns <- 150 
    }else{
      turns <- 300
    }
  }
  if(problemName == "dbpedia"){
    turns <- 700
  }
  if(problemName == "kickstarter"){
    turns <- 700
  }
  if(problemName == "wikipedia"){
    turns <- 400
  }
  if(problemName == "blogcatalog"){
    turns <- 1400
  }
  if(problemName == "flickr"){
    turns <- 4000
  }
  if(problemName == "ppi"){
    turns <- 2000
  }
  if(problemName == "cora"){
    turns <- 1100
  }
  if(problemName == "lj"){
    turns <- 1200
  }
  if(problemName == "dblp"){
    turns <- 1200
  }
  if(problemName == "amazon"){
    turns <- 2000
  }
  return(turns)
}
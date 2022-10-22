#-- karen
#tpine-locality - tpine

load_TargetLine <- function(problemName){
  
  if(problemName == "citeseer"){
    tline <- 2
  }
  if(problemName == "donors"){
    tline <- 284
  }
  if(problemName == "dbpedia"){
    tline <- 0
  }
  if(problemName == "kickstarter"){
    tline <- 180
  }
  if(problemName == "wikipedia"){
    tline <- 48
  }
  if(problemName == "blogcatalog"){
    tline <- 7
  }
  if(problemName == "flickr"){
    tline <- 148
  }
  if(problemName == "ppi"){
    tline <- 32
  }
  if(problemName == "cora"){
    tline <- 34
  }
  if(problemName == "lj"){
    tline <- 4915
  }
  if(problemName == "dblp"){
    tline <- 4972
  }
  if(problemName == "amazon"){
    tline <- 4832
  }
  return(tline)
}
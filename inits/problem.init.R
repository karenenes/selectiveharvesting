source("problem.R")

PROBLEMS <- list(

    dbpedia = Problem$new("dbpedia",
        "../../../data/datasets/wang2013/dbpedia/done/dbpedia.ungraph.txt.gz",
        "../../../data/datasets/wang2013/dbpedia/done/dbpedia.attribs.txt.gz",
        accept_trait = 0, target_size = 725, nattribs = 5),
    dbpedia2 = Problem$new("dbpedia2",
        "../../../data/datasets/wang2013/dbpedia/done/dbpedia.ungraph.txt.gz",
        "../../../data/datasets/wang2013/dbpedia/done/dbpedia.attribs.txt.gz",
        accept_trait = 3, target_size = 2633, nattribs = 5),
    dbpedia3 = Problem$new("dbpedia3",
        "../../../data/datasets/wang2013/dbpedia/done/dbpedia.ungraph.txt.gz",
        "../../../data/datasets/wang2013/dbpedia/done/dbpedia.attribs.txt.gz",
        accept_trait = 2, target_size = 1398, nattribs = 5),

    citeseer = Problem$new("citeseer",
        "../../../data/datasets/gccs/citeseer/labeled/citeseer.gcc.txt.gz",
        "../../../data/datasets/gccs/citeseer/labeled/citeseer.att.txt.gz",
        accept_trait = 2, target_size = 1583, nattribs = 10),

    citeseer2 = Problem$new("citeseer2",
        "../../../data/datasets/gccs/citeseer/labeled/citeseer.gcc.txt.gz",
        "../../../data/datasets/gccs/citeseer/labeled/citeseer.att.txt.gz",
        accept_trait = 5, target_size = 1576, nattribs = 10),

    citeseer3 = Problem$new("citeseer3",
        "../../../data/datasets/gccs/citeseer/labeled/citeseer.gcc.txt.gz",
        "../../../data/datasets/gccs/citeseer/labeled/citeseer.att.txt.gz",
        accept_trait = 0, target_size = 1517, nattribs = 10),

    wikipedia = Problem$new("wikipedia",
        "../../../data/datasets/wang2013/wikipedia/done/wikipedia.ungraph.txt.gz",
        "../../../data/datasets/wang2013/wikipedia/done/wikipedia.attribs.txt.gz",
        accept_trait = 48, target_size = 202, nattribs = 93),
    wikipedia2 = Problem$new("wikipedia2",
        "../../../data/datasets/wang2013/wikipedia/done/wikipedia.ungraph.txt.gz",
        "../../../data/datasets/wang2013/wikipedia/done/wikipedia.attribs.txt.gz",
        accept_trait = 49, target_size = 550, nattribs = 93),
    wikipedia3 = Problem$new("wikipedia3",
        "../../../data/datasets/wang2013/wikipedia/done/wikipedia.ungraph.txt.gz",
        "../../../data/datasets/wang2013/wikipedia/done/wikipedia.attribs.txt.gz",
        accept_trait = 0, target_size = 238, nattribs = 93),

    # --------------------------------------------------------------------------

    flickr1 = Problem$new("flickr1",
        "../../../data/datasets/gccs/flickr/flickr.gcc.txt.gz",
        "../../../data/datasets/gccs/flickr/flickr.att.txt.gz",
        accept_trait = 178, target_size = 6240, nattribs = 195),
        
    youtube = Problem$new("youtube",
        "../../../data/datasets/gccs/youtube/youtube.gcc.txt.gz",
        "../../../data/datasets/gccs/youtube/youtube.att.txt.gz",
        accept_trait = 13, target_size = 3532, nattribs = 47),
    
    blogcatalog1 = Problem$new("blogcatalog1",
        "../../../data/datasets/gccs/blogcatalog/blogcatalog.gcc.txt.gz",
        "../../../data/datasets/gccs/blogcatalog/blogcatalog.att.txt.gz",
        accept_trait = 10, target_size = 986, nattribs = 39),
    
    # ---------------------------------------------------------------------------

    donors = Problem$new("donors", 
        "../../../data/datasets/gccs/donors/donors.gcc.txt.gz",
        "../../../data/datasets/gccs/donors/donors.att.txt.gz",
        accept_trait = 284, target_size = 56, nattribs = 285),
    donors2 = Problem$new("donors2", 
        "../../../data/datasets/gccs/donors/donors.gcc.txt.gz",
        "../../../data/datasets/gccs/donors/donors.att.txt.gz",
        accept_trait = 285, target_size = 39, nattribs = 287),
    donors3 = Problem$new("donors3", 
        "../../../data/datasets/gccs/donors/donors.gcc.txt.gz",
        "../../../data/datasets/gccs/donors/donors.att.txt.gz",
        accept_trait = 286, target_size = 38, nattribs = 287),

    dblp = Problem$new("dblp",
        "../../../data/datasets/dblp/com-dblp.ungraph.txt.gz",
        "../../../data/datasets/dblp/com-dblp.top5000.cmty.txt.gz",
        accept_trait = 4972, target_size = 7556, nattribs = 5000),
    
    amazon1 = Problem$new("amazon1",
        "../../../data/datasets/amazon/com-amazon.ungraph.txt.gz",
        "../../../data/datasets/amazon/com-amazon.top5000.cmty.txt.gz",
        accept_trait = 4832, target_size = 328, nattribs = 5000),
    
    lj = Problem$new("lj",
        "../../../data/datasets/lj/com-lj.ungraph.txt.gz",
        "../../../data/datasets/lj/com-lj.top5000.cmty.txt.gz",
        accept_trait = 4915, target_size = 1441, nattribs = 5000),
    
    kickstarter = Problem$new("kickstarter",
        "../../../data/datasets/kickstarter/kickstarter.ungraph.txt.gz",
        "../../../data/datasets/kickstarter/kickstarter.attribs.txt.gz",
        accept_trait = 180, target_size = 1457, nattribs = 183),
    kickstarter2 = Problem$new("kickstarter2",
        "../../../data/datasets/kickstarter/kickstarter.ungraph.txt.gz",
        "../../../data/datasets/kickstarter/kickstarter.attribs.txt.gz",
        accept_trait = 181, target_size = 1260, nattribs = 183),
    kickstarter3 = Problem$new("kickstarter3",
        "../../../data/datasets/kickstarter/kickstarter.ungraph.txt.gz",
        "../../../data/datasets/kickstarter/kickstarter.attribs.txt.gz",
        accept_trait = 182, target_size = 854, nattribs = 183),
    
    # ---------------------------------------------------------------------------
    # novos -- TANE
    blogcatalog = Problem$new("blogcatalog", 
                         "../../../data/datasets/new/blogcatalog/blogcatalog.gcc.txt.gz",
                         "../../../data/datasets/new/blogcatalog/blogcatalog.att.txt.gz",
                         accept_trait = 7, target_size = 1423, nattribs = 39),
    
    flickr = Problem$new("flickr", 
                               "../../../data/datasets/new/flickr/flickr.gcc.txt.gz",
                               "../../../data/datasets/new/flickr/flickr.att.txt.gz",
                               accept_trait = 148, target_size = 9802, nattribs = 195),
    
    ppi = Problem$new("ppi", 
                          "../../../data/datasets/new/ppi/ppi.gcc.txt.gz",
                          "../../../data/datasets/new/ppi/ppi.att.txt.gz",
                          accept_trait = 32, target_size = 3048, nattribs = 121),
    
    cora = Problem$new("cora", 
                      "../../../data/datasets/new/cora/cora.gcc.txt.gz",
                      "../../../data/datasets/new/cora/cora.att.txt.gz",
                      accept_trait = 34, target_size = 1089, nattribs = 70),
    
    amazon = Problem$new("amazon",
                          "../../../data/datasets/new/amazon/amazon.gcc.txt.gz",
                          "../../../data/datasets/new/amazon/amazon.att.txt.gz",
                          accept_trait = 4832, target_size = 328, nattribs = 5000)
    
    
)


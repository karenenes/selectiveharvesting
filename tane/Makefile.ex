MAIN = node2vec
DEPH = $(EXSNAPADV)/n2v.h $(EXSNAPADV)/word2vec.h $(EXSNAPADV)/biasedrandomwalk.h $(EXSNAPADV)/dyn_util.h
DEPCPP = $(EXSNAPADV)/n2v.cpp $(EXSNAPADV)/word2vec.cpp $(EXSNAPADV)/biasedrandomwalk.cpp $(EXSNAPADV)/dyn_util.cpp
CXXFLAGS += $(CXXOPENMP)

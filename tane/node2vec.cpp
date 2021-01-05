#include "stdafx.h"

#include "n2v.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "dyn_util.h"

void ParseArgs(int& argc, char* argv[], TStr& InFile, TStr& OutFile,
 int& Dimensions, int& WalkLen, int& NumWalks, int& WinSize, int& Iter,
 bool& Verbose, double& ParamP, double& ParamQ, bool& Directed, bool& Weighted,
 bool& OutputWalks, bool &Dynamic, TStr& PrevWalksFile,
 TStr& PrevWeightsFile, int& CurrentTurn, int& TotalTurns) {
  Env = TEnv(argc, argv, TNotify::StdNotify);
  Env.PrepArgs(TStr::Fmt(
    "\nAn algorithmic framework for representational learning on graphs."));
  InFile = Env.GetIfArgPrefixStr("-i:", "graph/karate.edgelist",
   "Input graph path");
  OutFile = Env.GetIfArgPrefixStr("-o:", "emb/karate.emb",
   "Output graph path");
  Dimensions = Env.GetIfArgPrefixInt("-d:", 128,
   "Number of dimensions. Default is 128");
  WalkLen = Env.GetIfArgPrefixInt("-l:", 80,
   "Length of walk per source. Default is 80");
  NumWalks = Env.GetIfArgPrefixInt("-r:", 10,
   "Number of walks per source. Default is 10");
  WinSize = Env.GetIfArgPrefixInt("-k:", 10,
   "Context size for optimization. Default is 10");
  Iter = Env.GetIfArgPrefixInt("-e:", 1,
   "Number of epochs in SGD. Default is 1");
  ParamP = Env.GetIfArgPrefixFlt("-p:", 1,
   "Return hyperparameter. Default is 1");
  ParamQ = Env.GetIfArgPrefixFlt("-q:", 1,
   "Inout hyperparameter. Default is 1");
  Verbose = Env.IsArgStr("-v", "Verbose output.");
  Directed = Env.IsArgStr("-dr", "Graph is directed.");
  Weighted = Env.IsArgStr("-w", "Graph is weighted.");
  OutputWalks = Env.IsArgStr("-ow",
    "Output random walks instead of embeddings.");
  // Dynamic node2vec parameters
  CurrentTurn = Env.GetIfArgPrefixInt("-ct:", 0,
   "Current turn of the dynamic network exploration. Default is 0");
  TotalTurns = Env.GetIfArgPrefixInt("-tt:", 0,
   "Total number of turns of network exploration (budget). Default is 0");
  Dynamic = Env.IsArgStr("-dy", "Dynamic node2vec.");
  PrevWalksFile = Env.GetIfArgPrefixStr("-pwa:", "",
    "Load previous iterations' walks.");
  PrevWeightsFile = Env.GetIfArgPrefixStr("-pwe:", "",
    "Load previous iterations' weights.");
}

void ReadGraph(TStr& InFile, bool& Directed, bool& Weighted, bool& Verbose, PWNet& InNet) {
  TFIn FIn(InFile);
  int64 LineCnt = 0;
  try {
    while (!FIn.Eof()) {
      TStr Ln;
      FIn.GetNextLn(Ln);
      TStr Line, Comment;
      Ln.SplitOnCh(Line,'#',Comment);
      TStrV Tokens;
      Line.SplitOnWs(Tokens);
      if(Tokens.Len()<2){ continue; }
      int64 SrcNId = Tokens[0].GetInt();
      int64 DstNId = Tokens[1].GetInt();
      double Weight = 1.0;
      if (Weighted) { Weight = Tokens[2].GetFlt(); }
      if (!InNet->IsNode(SrcNId)){ InNet->AddNode(SrcNId); }
      if (!InNet->IsNode(DstNId)){ InNet->AddNode(DstNId); }
      InNet->AddEdge(SrcNId,DstNId,Weight);
      if (!Directed){ InNet->AddEdge(DstNId,SrcNId,Weight); }
      LineCnt++;
    }
    if (Verbose) { printf("Read %lld lines from %s\n", (long long)LineCnt, InFile.CStr()); }
  } catch (PExcept Except) {
    if (Verbose) {
      printf("Read %lld lines from %s, then %s\n", (long long)LineCnt, InFile.CStr(),
       Except->GetStr().CStr());
    }
  }
}

void WriteWalks(TFOut &FOut, TVVec<TInt, int64>& WalksVV){
  // Write dimensions to file
  FOut.PutInt(WalksVV.GetXDim());
  FOut.PutCh(' ');
  FOut.PutInt(WalksVV.GetYDim());
  FOut.PutLn();
  // Write walks
  for (int64 i = 0; i < WalksVV.GetXDim(); i++) {
    for (int64 j = 0; j < WalksVV.GetYDim(); j++) {
      FOut.PutInt(WalksVV(i,j));
    	if(j+1==WalksVV.GetYDim()) {
              FOut.PutLn();
    	} else {
              FOut.PutCh(' ');
    	}
    }
  }
}

bool ReadWalks(TStr& InFile, TVVec<TInt, int64>& WalksVV, bool& Verbose){
  TFIn FIn(InFile);
  bool First = 1;
  int64 LineCnt = 0;
  int64 NodeCount = 0;
  int64 EmbeddingSize = 0;
  try {
    while(!FIn.Eof()) {
      TStr Ln;
      FIn.GetNextLn(Ln);
      TStr Line, Comment;
      Ln.SplitOnCh(Line,'#',Comment);  // Remove comments
      TStrV Tokens;
      Line.SplitOnWs(Tokens);  // Split to vector of tokens
      if(Tokens.Len()<2){
        printf("Line %lld contains less than two tokens.\n",
          (long long)LineCnt);
        continue;
      }
      if(First){
        int64 DimX, DimY;
        DimX = Tokens[0].GetInt();
        DimY =Tokens[1].GetInt();
        WalksVV = TVVec<TInt, int64>(DimX, DimY);
        First = 0;
      }else{
        for(int64 i = 0; i < Tokens.Len(); i++){
          // LineCnt - 1 will be the matrix's row
          WalksVV.PutXY(LineCnt - 1, i, Tokens[i].GetInt());
        }
      }
      LineCnt++; // Counts how many lines of the file have been read
    }
    if (Verbose) {
      printf("Read %lld lines for (%lld nodes) from %s\n",
        (long long)LineCnt,
        (long long)NodeCount,
        InFile.CStr());
    }
  } catch (PExcept Except) {
    if (Verbose) {
      printf("ERROR: Read %lld lines from %s, then %s\n",
        (long long)LineCnt, InFile.CStr(),
       Except->GetStr().CStr());
    }
    return false;
  }
  return true;;
}

void WriteEmbeddings(TFOut &FOut, TIntFltVH& EmbeddingsHV){
  bool First = 1;
  for (int i = EmbeddingsHV.FFirstKeyId(); EmbeddingsHV.FNextKeyId(i);) {
    if (First) {
      FOut.PutInt(EmbeddingsHV.Len());
      FOut.PutCh(' ');
      FOut.PutInt(EmbeddingsHV[i].Len());
      FOut.PutLn();
      First = 0;
    }
    FOut.PutInt(EmbeddingsHV.GetKey(i));
    for (int64 j = 0; j < EmbeddingsHV[i].Len(); j++) {
      FOut.PutCh(' ');
      FOut.PutFlt(EmbeddingsHV[i][j]);
    }
    FOut.PutLn();
  }
}

bool ReadEmbeddings(TStr& InFile, TIntFltVH& EmbeddingsHV, bool& Verbose){
  TFIn FIn(InFile);
  bool First = 1;
  int64 LineCnt = 0;
  int64 NodeCount = 0;
  int64 EmbeddingSize = 0;
  try {
    while(!FIn.Eof()) {
      TStr Ln;
      FIn.GetNextLn(Ln);
      TStr Line, Comment;
      Ln.SplitOnCh(Line,'#',Comment);  // Remove comments
      TStrV Tokens;
      Line.SplitOnWs(Tokens);  // Split to vector of tokens
      if(Tokens.Len()<2){
        printf("Line %lld contains less than two tokens.\n",
          (long long)LineCnt);
        continue; }
      if(First){
        NodeCount = Tokens[0].GetInt();
        EmbeddingSize = Tokens[1].GetInt();
        First = 0;
      }else{
        // Read node identifier
        int64 NId = Tokens[0].GetInt();
        TFltV Weights(EmbeddingSize);
        for(int64 i = 0; i < EmbeddingSize; i++){
          // Try to read weight from file
          TFlt weight;
          try{
            weight = Tokens[i + 1].GetFlt();
          }catch (PExcept Except) {
            if (Verbose) {
              printf(
                "ERROR: FAILED in %lld th dimension of emb. for node %lld\n %s\n",
                (long long)i, (long long)NId,
               Except->GetStr().CStr());
            }
            return false;
          }
          // Set on vector
          Weights[i] = weight;
        }
        // Add to hash table
        EmbeddingsHV.AddDat(NId, Weights);
      }
      LineCnt++;
    }
    if (Verbose) {
      printf("Read %lld lines for (%lld nodes) from %s\n",
        (long long)LineCnt,
        (long long)NodeCount,
        InFile.CStr());
    }
  } catch (PExcept Except) {
    if (Verbose) {
      printf("ERROR: Read %lld lines from %s, then %s\n",
        (long long)LineCnt, InFile.CStr(),
       Except->GetStr().CStr());
    }
    return false;
  }
  return true;
}

void WriteOutput(TStr& OutFile, TIntFltVH& EmbeddingsHV, TVVec<TInt,
  int64>& WalksVV, bool& OutputWalks, bool &Dynamic) {
  // printf("Called WriteOutput, OutFile: %s\n", OutFile.CStr());
  TStr WalksPath("");
  if(Dynamic){
    WalksPath += OutFile;
    WalksPath += ".walk";
  }
  TFOut FOut(OutFile);
  if (!Dynamic && OutputWalks) {
    WriteWalks(FOut, WalksVV);
    return;
  }else if(Dynamic) { // Have to write walks as output
    TFOut WalksOut(WalksPath);
    WriteWalks(WalksOut, WalksVV);
  }
  WriteEmbeddings(FOut, EmbeddingsHV);
}

int main(int argc, char* argv[]) {
  TStr InFile,OutFile;
  int Dimensions, WalkLen, NumWalks, WinSize, Iter;
  double ParamP, ParamQ;
  bool Directed, Weighted, Verbose, OutputWalks;
  
  // Additional variables for dynamic node2vec
  int CurrentTurn = 0;
  int TotalTurns = 0;
  bool Dynamic = false;
  TStr PrevWalksFile("");
  TStr PrevWeightsFile("");
  // END additional variables for dynamic node2vec

  ParseArgs(argc, argv, InFile, OutFile, Dimensions, WalkLen,
            NumWalks, WinSize, Iter, Verbose, ParamP, ParamQ,
            Directed, Weighted, OutputWalks, Dynamic, PrevWalksFile,
            PrevWeightsFile, CurrentTurn, TotalTurns);

  PWNet InNet = PWNet::New();
  TIntFltVH EmbeddingsHV;
  TIntFltVH PrevEmbeddingsHV;       // Load previous iteration's embeddings
  TVVec <TInt, int64> WalksVV;
  TVVec <TInt, int64> PrevWalksVV;  // Load previous iteration's walks
  if(Verbose) {
    printf("Initial PrevWalksVV size: X: %ld   Y: %ld \n",
           PrevWalksVV.GetXDim(),
           PrevWalksVV.GetYDim());
  }

  if(Dynamic){
    if(Verbose) printf("Dynamic node2vec activated!!\n");
    // Check if everything is okay with the parameters before starting
    bool input_conditions = CheckInputOutputConditions(InFile,
      PrevWalksFile, PrevWeightsFile,
      TotalTurns, Verbose);
    if(!input_conditions) return -1;
    // BEGIN Load Previous Iterations Walks
    if(!PrevWalksFile.Empty()){
      if(Verbose){
        printf("Size of previous iteration walks BEFORE READING: %ld %ld\n",
          PrevWalksVV.GetXDim(), PrevWalksVV.GetYDim());
      }
      if(!ReadWalks(PrevWalksFile, PrevWalksVV, Verbose)){
        printf("ERROR reading previous iterations walks, on file: %s\n",
          PrevWalksFile.CStr());
        return -1;
      }
      if(Verbose){
        printf("Size of previous iteration walks AFTER READING: %ld %ld\n",
          PrevWalksVV.GetXDim(), PrevWalksVV.GetYDim());
      }
    }
    // END Load Previous Iterations Walks

    // BEGIN Load Previous Iterations Weights
    if(!PrevWeightsFile.Empty()){
      if(Verbose){
        printf("Size of previous iteration weights BEFORE READING: %d\n",
          PrevEmbeddingsHV.Len());
      }
      if(!ReadEmbeddings(PrevWeightsFile, PrevEmbeddingsHV, Verbose)){
        // Something went wrong.
        return -1;
      }
      if(Verbose){
        printf("Size of previous iteration weights AFTER READING: %d\n",
          PrevEmbeddingsHV.Len());
      }
    }
    // END Load Previous Iterations Weights

    // BEGIN Adapt P and Q parameters
    if(Verbose) printf("P and Q BEFORE ADAPTING: %.6f %.6f\n", ParamP, ParamQ);
    SetParamsPandQ(ParamP, ParamQ, CurrentTurn, TotalTurns);
    if(Verbose) printf("P and Q AFTER ADAPTING: %.6f %.6f\n", ParamP, ParamQ);
    // END Adapt P and Q parameters
    bool LoadPrevWalks = !PrevWalksVV.Empty();
    bool LoadPrevWeights = !PrevEmbeddingsHV.Empty();
    ReadGraph(InFile, Directed, Weighted, Verbose, InNet);
    node2vec(InNet,
      ParamP,
      ParamQ,
      Dimensions,
      WalkLen,
      NumWalks,
      WinSize,
      Iter,
      Verbose,
      OutputWalks,
      Dynamic,            // Included this parameter
      LoadPrevWalks,      // Included this parameter
      LoadPrevWeights,    // Included this parameter
      WalksVV,
      PrevWalksVV,        // Included this parameter
      EmbeddingsHV,
      PrevEmbeddingsHV);  // Included this parameter
    WriteOutput(OutFile, EmbeddingsHV, WalksVV, OutputWalks, Dynamic);
  }else{
    ReadGraph(InFile, Directed, Weighted, Verbose, InNet);
    node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize, Iter, 
     Verbose, OutputWalks, WalksVV, EmbeddingsHV);
    WriteOutput(OutFile, EmbeddingsHV, WalksVV, OutputWalks, Dynamic);    
  }
  return 0;
}

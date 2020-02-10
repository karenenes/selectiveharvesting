# get performance for each method on each dataset at a given timepoint
options(error=recover)

#method2dir=list(
#                pnb='pnb_all',
#                sn_ucb1='sn_ucb1_all',
#                mod='rr_max_mod_all',
#                activesearch='rr_max_activesearch_all',
#                logistic='rr_max_logistic_all',
#                ewrls='rr_max_ewrls_all',
#                svm='rr_max_svm_all',
#                rforest='rr_max_rforest.p_all',
#                listnet='rr_max_listnet_all',
#                rr1='rr_max_config1_all',
#                dts1='dts.5_max_config2_all'
#)
method2dir=list(
#                mod='rr_max_mod_all',
#                activesearch='rr_max_activesearch_all',
#                svm='rr_max_svm_all',
#                rforest='rr_max_randomforest.p_all',
#                listnet='rr_max_listnet_all',
#                rr1='rr_max_config2_all',
#                dts1='dts.5_max_config2_all',
                glm = 'rr_max_GLM_all'
)

#method2str=list(
#                pnb='PNB',
#                sn_ucb1='SN-UCB1',
#                mod='MOD \\cmark',
#                activesearch='Active Search \\cmark',
#                logistic='Logistic Regression \\cmark',
#                ewrls='EWLS',
#                svm='SV Regression',
#                rforest='Random Forest \\cmark',
#                listnet='ListNet \\cmark',
#                rr1='Round-Robin (all \\cmark)',
#                dts1='\\ALGO (all \\cmark)'
#)
method2str=list(
#                mod='MOD',
#                activesearch='Active Search',
#                svm='SV Regression',
#                rforest='Random Forest',
#                listnet='ListNet',
#                rr1='Round-Robin',
#                dts1='\\ALGO',
                glm = 'Generalized Linear Model'

)



# budget = 100*floor(ntargets*multiplier/100), multiplier is
# 1.0, 1.0, 2.0, 2.0, 1.0, 0.17, 0.83
#dataset2time=list(citeseer=1500,
#                  dbpedia=700,
#                  wikipedia=400,
#                  donors=100,
#                  kickstarter=700,
#                  dblp=1200,
#                  lj=1200,
#                  citeseer2=1500,
#                  citeseer3=1500,
#                  dbpedia2=700,
#                  dbpedia3=700,
#                  wikipedia2=400,
#                  wikipedia3=400,
#                  donors2=100,
#                  donors3=100,
#                  kickstarter2=700,
#                  kickstarter3=700
#)
dataset2time=list(citeseer=1500,
                  dbpedia=700,
                  wikipedia=400,
                  donors=100,
                  kickstarter=700
#                  dblp=1200,
#                  lj=1200
)



#dataset2str=list(citeseer='{\\bf CS}',
#                  dbpedia='{\\bf DBP}',
#                  wikipedia='{\\bf WK}',
#                  donors='{\\bf DC}',
#                  kickstarter='{\\bf KS}',
#                  dblp='{\\bf DBL}',
#                  lj='{\\bf LJ}'
#)

ntargets=list(citeseer=1583,
              dbpedia=725,
              wikipedia=202,
              donors=56,
              kickstarter=1457,
              dblp=7556,
              lj=1441
)

#methods=c(
#          'pnb',
#          'sn_ucb1',
#          'mod',
#          'activesearch',
#          'logistic',
#          'ewrls',
#          'svm',
#          'rforest',
#          'listnet',
#          'rr1',
#          'dts1'
#)
methods=c(
#          'mod',
#          'activesearch',
#          'svm',
#          'rforest',
#          'rr1',
#          'dts1',
          'glm'
)

# read path to results
args = commandArgs(trailingOnly=TRUE)

# test if there one argument: if not, return an error
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} else {
  # default output file
  path = args[1]
}


means = matrix(NA,nrow=length(methods),length(dataset2time))
rownames(means)=methods
colnames(means)=names(dataset2time)
sds = means

for(method in methods) {
    for(dataset in names(dataset2time)) {
        filename = Sys.glob(sprintf('%s/%s*/extracted/%s*.tsv', path, method2dir[[method]], dataset))
        print(filename)
        z = read.table(filename)
        means[method,dataset] = z$V1[dataset2time[[dataset]]]
        sds[method,dataset] = z$V2[dataset2time[[dataset]]]
    }
}



#######################################################################################################
# Paper's table generators below.
#######################################################################################################

library(xtable)
 
# sds[is.na(sds)] = 0
# 
# # generate table 5
# top.k = c(5,3,1)
# ratio = matrix(NA,nrow=length(dataset2time),ncol=2*length(top.k))
# rownames(ratio) = names(dataset2time)
# 
# rows=nrow(means)
# cols=ncol(means)
# standalone <- means[1:(rows-2),1:cols]
# mab <- means[(rows-1):rows,1:cols]
# 
# for(dataset in names(dataset2time)) {
#     z <- sort(standalone[,dataset],decreasing=TRUE)
#     avg.perf <- c(mean(z[1:5]),mean(z[1:3]),z[1])
#     zz <- round(matrix(t(matrix(rep(mab[,dataset],3),3,2,byrow=TRUE)/avg.perf),nrow=1),digits=2)
#     ratio[dataset,] <- zz
# }
# x <- xtable(ratio)
# print(x)
# 
# 
# # print datasets timepoints and percentage that DTS1 collected
# times = unlist(dataset2time)
# times = cbind(times,means[methods[length(methods)],]/unlist(ntargets),times/unlist(ntargets))
# colnames(times) = c('budget','% found','budget/ntargets')
# times = round(times,digits=2)
# print(times)
# 
# # for each entry check if DTS is: better*, better, worse, worse*
# comp = matrix('',nrow=length(methods),length(dataset2time))
# rownames(comp)=methods[1:(length(methods))]
# colnames(comp)=names(dataset2time)
# for(method in methods) {
#     #if(method == 'dts1')
#     #    break
#     for(dataset in names(dataset2time)) {
#         if(means['dts1',dataset] < means[method,dataset]) {
#             if(means['dts1',dataset]+1.96*sds['dts1',dataset]/sqrt(80) <
#                means[method,dataset]-1.96*sds[method,dataset]/sqrt(80)) {
# 		    #comp[method,dataset] <- 'better*'
# 		    comp[method,dataset] <- '$^+$'
#             } else {
# 		    #comp[method,dataset] <- 'better'
# 		    comp[method,dataset] <- ''
#             }
#         } else {
#             if(means[method,dataset]+1.96*sds[method,dataset]/sqrt(80) <
#                means['dts1',dataset]-1.96*sds['dts1',dataset]/sqrt(80)) {
# 		    #comp[method,dataset] <- 'worse*'
# 		    comp[method,dataset] <- '$^*$'
#             } else {
# 		    #comp[method,dataset] <- 'worse'
# 		    comp[method,dataset] <- ''
#             }
#         }
#     }
# }
#print(comp)
 
 
 max.inds <- matrix(NA,nrow=2,ncol=ncol(means))
 
 #find the top 2 in each column
 for(jdx in 1:ncol(means))
     max.inds[,jdx] <- sort(means[,jdx],decreasing=T,index.return=T)$ix[1:2]
 
 # normalize means by the best
 normalize=F
 if(normalize) {
     norm = apply(means[1:(nrow(means)-2),],MARGIN=2,FUN=max)
     means = t(t(means)/norm)
     means.str = format(means,digits=3,nsmall=2)
 } else {
     means.str = format(means,digits=3,nsmall=1)
 }
 
 means.str = matrix(paste(means.str,comp,sep=''),nrow=nrow(means.str),ncol=ncol(means.str),
                   dimnames=dimnames(means.str))
 
 # put top 2 of each column in bold
 for(jdx in 1:ncol(means.str)) {
     for(idx in max.inds[,jdx]) {
         means.str[idx,jdx] = sprintf('{\\bf %s}', means.str[idx,jdx])
     }
 }
 
 rownames(means.str)=unlist(Map(function(x) method2str[[x]], rownames(means.str)))
 
 print(means.str)
 x <- xtable(means.str)
 print(x,sanitize.text.function=function(x){x},floating.environment='table*')#,
#      hline.after=c(4,9))

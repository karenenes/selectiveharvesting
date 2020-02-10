options(error=recover)
source('plotMany.R')
#options(warn=2)

#cfg2name <- list(
#'11111'='Round-Robin',
#'10000'='MOD',
#                 '01000'='Active Search',
#                 '00100'='Logistic',
#                 '00010'='Random Forest',
#                 '00001'='ListNet'#,
#                 )
cfg2name <- list(
'config2'='Round-Robin',
'mod'='MOD',
'activesearch'='Active Search',
#'logistic'='Logistic',
'svm'='SV Regression',
'randomforest.p'='Random Forest',
'listnet'='ListNet'#,
)

dataset2name <- list(
'donors'='DonorsChoose',
'citeseer'='CiteSeer: NIPS papers',
'wikipedia'='Wikipedia: Obj. Orient. Prog.',
'dbpedia'='DBpedia: admin. regions',
'lj'='LiveJournal',
'dblp'='DBLP',
'kickstarter'='Kickstarter: DFA project'
)

#other2name <- list('b0.99_l1.0'='EWLS',
#                   'neville'='PNB',
#                   'SN_UCB1'='SN-UCB1',
#                   'svm-mod'='SV Regression')


# read path to results
args = commandArgs(trailingOnly=TRUE)

# test if there one argument: if not, return an error
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} else {
  # default output file
  path = args[1]
}

#for(dataset in c('citeseer','kickstarter','dbpedia','donors','wikipedia','dblp','lj')) {
for(dataset in c('citeseer','kickstarter','dbpedia','donors','wikipedia')) {
   filenames = c()
   keys = c()
   colors = c()
   ltypes = c()
   for(idx in 1:length(cfg2name)) {
       config = names(cfg2name)[idx]
       filenames = c(filenames,Sys.glob(sprintf('%s/rr_max_%s_all/extracted/%s_t*_a*_m*_s*_*.npositive.RData',path, config,dataset)))
       keys = c(keys,cfg2name[[config]])
       #colors = c(colors,idx)
       ltypes = c(ltypes,1)
   }
   filenames = c(filenames,Sys.glob(sprintf('%s/dts.5_max_config2_all/extracted/%s_t*_a*_m*_s*_*.npositive.RData',path, dataset)))
   keys = c(keys,expression(paste(D^3,'TS')))
   colors = c('#0000ff','#000000','#984ea3','#377eb8','#4daf4a','#e41a1c','#ff7f00')
   ltypes = c(ltypes,2)

   print(filenames)
   plotMany(filenames,keys,sprintf('%s/%s-line-%s.pdf',path,dataset,'Five'),colors,ltypes,
             normalize=T,plot.CI=T,titlename=dataset2name[[dataset]],show.legend=T)

   # as in the paper
   #plotMany(filenames,keys,sprintf('%s/norm.%s-line-%s.pdf',path,dataset,'Five'),colors,ltypes,
   #          normalize=T,plot.CI=T,titlename=dataset2name[[dataset]],
   #          ylims=c(0.88,1.12),
   #          hide.ylab=(dataset=='dblp'||dataset=='kickstarter'||dataset=='donors'),
   #          hide.xlab=(dataset=='dblp'||dataset=='dbpedia'||dataset=='donors'),
   #          show.legend=(dataset=='dbpedia'||dataset=='citeseer')
   #)

   # without normalizing
   #plotMany(filenames,keys,sprintf('%s/%s-line-%s.pdf',path,dataset,'All'),colors,ltypes,normalize=F,plot.CI=F)
}

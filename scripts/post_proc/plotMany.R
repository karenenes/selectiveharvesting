library(Hmisc)

timepoints <- list(
                   donors=300,
                   kickstarter=1000,
                   citeseer=1400,
                   dbpedia=1000,
                   wikipedia=1000,
                   dblp=1000,
                   lj=1000
                   )

plotlinerr_bars <- function(x,y,ystd,colory,ltypey,runs=50) {
  cinf95 = 1.96 * ystd/sqrt(runs)
  confIntUp = y + cinf95
  confIntDown = y - cinf95
  c_vec = t(col2rgb(colory)/255)
  colorerr = rgb(red=c_vec[1],green=c_vec[2],blue=c_vec[3],alpha=0.3)
  polygon(c(x,rev(x)),c(confIntUp,rev(confIntDown)),col=colorerr,border=NA)
  #lines(x,y,col=colory,lty=ltypey,lw=2)
}

getStats <- function(filename) {
  z <- load(filename)
  data <- do.call(rbind, get(z))
  return (list(means=apply(data,MARGIN=2,FUN=mean,na.rm=TRUE),
               sds=apply(data,MARGIN=2,FUN=sd,na.rm=TRUE),
               n=nrow(data)))
}



##
 # For each RData file given as input, plot average.
 ##
plotMany <- function (filenames, keys=NA, outfile=NA, colors=c(), ltypes=c(), titlename="",
                      plot.CI=F, normalize=FALSE, ylims=c(),
                      show.legend=T, hide.xlab=F, hide.ylab=F) {
  #stopifnot(!is.na(keys))
  stopifnot(!is.na(outfile))

  if(is.null(colors))
    colors <- 1:length(filenames)
  if(is.null(ltypes))
    ltypes <- rep(1,length(filenames))

  means <- c()
  sds <- c()
  min.len <- Inf
  for(filename in filenames) {
    stats = getStats(filename)
    # adjust number of columns if needed
    min.len = min(min.len,length(stats$means))
    means <- rbind(means[,1:min.len],stats$means[1:min.len])
    sds <- rbind(sds[,1:min.len],stats$sds[1:min.len])
  }
  means <- means[,1:min.len]
  sds <- sds[,1:min.len]
  #print(cbind(means[,700],1.96*sds[,700]/sqrt(160)))

  ylabel <- '# targets found'
  if(normalize) {
     #m <- colMeans(means)
     m <- means[1,]
     means <- t(t(means)/m)
     sds <- t(t(sds)/m)
     ylabel <- sprintf('%s (norm. by %s)',ylabel,keys[1])
     #ylabel <- sprintf('%s (norm. by %s)',ylabel,'original')
  }
  if(hide.ylab)
      ylabel <- ''

  xlabel <- '# queried nodes (t)'
  if(hide.xlab)
      xlabel <- ''

  ## set legend position to topright for donors, lj, kickstarter
  #legend.pos <- 'bottomright'
  #if( length(grep("donors",filenames[1])) + length(grep("kickstarter",filenames[1])) + length(grep("lj",filenames[1])) > 0 )
  #    legend.pos <- 'topright'

  pdf(outfile,pointsize=18,height=7-0.2*hide.xlab,width=9-0.2*hide.ylab)
  #pdf(outfile,pointsize=24,height=7-0.2*hide.xlab,width=9-0.2*hide.ylab)
  par(mgp=c(2.5,1,0),mar=c(3.5-1.1*hide.xlab,3.5-1.5*hide.ylab,1.5,1.5),oma=c(0,0,0,0))

  # define ylims
  if(is.null(ylims)) {
      ylims <- range(means)
  }

  max.x <- length(means[1,])
  for(idx in 1:length(filenames)) {
    if(idx == 1) {
        plot(means[idx,],ylim=ylims, type='l',col=colors[idx], lty=ltypes[idx],lwd=2,
           xlab=xlabel,ylab=ylabel)
        title(main=titlename, cex.lab=1.5)
    } else {
        lines(means[idx,],col=colors[idx], lty=ltypes[idx], lwd=2)
    }
    if(plot.CI)
        plotlinerr_bars(1:max.x, means[idx,],sds[idx,],colory=colors[idx],ltypey=ltypes[idx],runs=stats$n)
  }

  # draw arrows if lowest point is off-the-charts
  min.mask <- rep(-Inf,max.x)
  min.mask[1:round(0.1*max.x)] <- Inf
  max.mask <- rep(Inf,max.x)
  max.mask[1:round(0.1*max.x)] <- -Inf
  y.n <- 20
  x.n <- 17
  for(idx in 1:length(filenames)) {
    # find minimizer and maximizer
    min.jdx <- which.min(pmax(min.mask,means[idx,]))
    max.jdx <- which.max(pmin(max.mask,means[idx,]))

    # if minimizer is off-the-chart
    if(means[idx,min.jdx] < 0.99*ylims[1]) {
        # plot arrows and text
        arrows(x0=min.jdx, x1=min.jdx, y0=(ylims[2]/y.n)+(ylims[1]*(y.n-1)/y.n), y1 = .992*ylims[1],
               lwd=3, length=.1, col=colors[idx])
        text(x=min.jdx, y=(ylims[2]/(y.n-5))+(ylims[1]*(y.n-6)/(y.n-5)),
             sprintf('%.2f',means[idx,min.jdx]),
             adj=c(0.5,0), col=colors[idx], cex=0.8)
        # update mask
        min.mask[max(1,round(min.jdx-max.x/x.n)):min(max.x,round(min.jdx+max.x/x.n))] <- Inf
    }
    if(means[idx,max.jdx] > 1.02*ylims[2]) {
        # plot arrows and text
        arrows(x0=max.jdx, x1=max.jdx, y0=(ylims[2]*(y.n-1)/y.n)+(ylims[1]/y.n), y1 = 1.008*ylims[2],
               lwd=3, length=.1, col=colors[idx])
        text(x=max.jdx, y=(ylims[2]*(y.n-6)/(y.n-5))+(ylims[1]/(y.n-5)),
             sprintf('%.2f',means[idx,max.jdx]),
             adj=c(0.5,1.0), col=colors[idx], cex=0.8)
        # update mask
        max.mask[max(1,round(max.jdx-max.x/x.n)):max(max.x,round(max.jdx+max.x/x.n))] <- -Inf
    }
  }


  if(show.legend) {
      legend.pos <- 'bottomright'
      if( normalize ) {
          N = nrow(means)
          z <- largest.empty(x=rep(1:max.x,N),y=as.vector(t(means)),width=0.15,height=0.15)
          if(z$y > 0.9)
              legend.pos <- 'topright'
      }
      #legend.pos <- 'bottomright'
      legend(legend.pos,legend=keys,lty=ltypes,lwd=2,col=colors,bty='n',cex=0.8)
  }
  dev.off()

  # compute stats for printing
  if(normalize) {
     avg.over.t <- rowMeans(means)
     #print(cbind(keys,avg.over.t,means[,ncol(means)]))
  }
}

gcloessm <- function(Xy, x='gc', y='coverage', z='',
                 samplesize = 5000000, 
                 mappability = 0.9,
                 rep= 1000, 
                 xoutlier = c(0.001, 1-0.001),
                 youtlier = c(0, 1- 0.005),
                 aoutlier = c(0, 1- 0.01)){
    Xy <- as.data.frame(Xy)
    Xy$valid <- TRUE
    Xy$valid[Xy[[y]] <= 0 | Xy[[x]] < 0] <- FALSE

    xrange <- quantile(Xy[Xy$valid, x], prob = xoutlier, na.rm = TRUE)
    yrange <- quantile(Xy[Xy$valid, y], prob = youtlier, na.rm = TRUE)

    Xy$ideal <- TRUE
    Xy$ideal[!Xy$valid |
              Xy[[y]] <=yrange[1] | Xy[[y]] > yrange[2] |
              Xy[[x]] < xrange[1] | Xy[[x]] > xrange[2] ] <- FALSE

    if(z %in% colnames(Xy)){ Xy$ideal[ Xy[[z]] < mappability ] <- FALSE }
    
    i <- seq(0, 1, by = 0.001)
    sindex <- which(Xy$ideal)
    select <- sample(sindex, min(length(sindex), samplesize))
    lmodel <- loess(Xy[select, y] ~ Xy[select, x], span = 0.03)
    fmodel <- loess(predict(lmodel, i) ~ i, span = 0.3)
    Xy$copy <- Xy[[y]] / predict(fmodel, Xy[[x]])

    ###add mappability
    if(z %in% colnames(Xy)){
      arange <- quantile(Xy$copy[Xy$valid], prob = aoutlier, na.rm = TRUE)
      sindex <- which(Xy$copy < arange[2])
      select <- sample(sindex, min(length(sindex), samplesize))
      mmodel <- approxfun(lowess(Xy[select, z], Xy[select, 'copy']))
      Xy$copy <- Xy$copy / mmodel(Xy[[z]])
    }

    Xy$copy[Xy$copy <= 0 | abs(Xy$copy) == Inf ] = NA
    Xy$logcopy <- log(Xy$copy, 2)
    return(Xy)
}

gclowessu <- function(Xy, x='gc', y='coverage'){
    xraw <- Xy[[x]]
    yraw <- Xy[[y]]
    NONAIdx  <- which(!is.na(xraw) & !is.na(yraw) & yraw > 0)
    
    xflt <- Xy[NONAIdx, x] 
    yflt <- Xy[NONAIdx, y] 
    ypre <- rep(NA, length(yraw))
    
    Md <- lowess(xflt, log2(yflt), f=0.3)
    Ma <- approx(Md$x, Md$y, xraw)
    Ly <- log2(Xy[[y]]) - Ma$y
    Ly[abs(Ly)==Inf] = NA
    Xy$copy <- 2^Ly
    Xy$logcopy <- Ly
    return(Xy)
}

segmentcbs <- function(df, copy='copy', chrom='chrom', pos='start', sid='c05296'){
  CNA.object <- CNA(df[copy],df[[chrom]],df[[pos]],
                    data.type="logratio",sampleid=sid)

  smoothed   <- smooth.CNA(CNA.object)
  segmented  <- segment(smoothed,
                      #undo.splits="sdundo", undo.SD=3, 
                      #alpha=, weights=
                      verbose=1)
  #segout <- cbind(segmented$output, segmented$segRows)
  segout <- segmented$output
  colnames(segout) <- c('SID', 'chrom', 'start', 'end', 'binsnum', 'meanlog2CN')
  segout <- segout[order(as.numeric(segout$chrom), segout$start),]
  return (segout)
}

segmenthmm <- function(logcopy){
  default_param <- HMMsegment(logcopy, getparam = TRUE)
  longseg_param <- default_param
  longseg_param$e <- 0.7
  longseg_param$strength <- 100
  longseg_segments <- HMMsegment(logcopy, longseg_param)
  print(longseg_segments)
}

plotCorr <- function(df, X='gc', yraw='reads', ycor='copy', points = 10000, ...) {
  par(mfrow = c(1, 2))
  plot(df[[X]], df[[yraw]],
       col = densCols(df[[X]], df[[yraw]]),
       pch = 20, cex=0.1, 
       ylab = "Uncorrected Readcount", xlab = "GC content",
       main = "GC Bias in Uncorrected Readcounts", ...)
  
  plot(df[[X]], df[[ycor]],
       col = densCols(df[[X]], df[[ycor]]),
       pch = 8, cex=0.1, 
       ylab  = "corrected Readcount", xlab = "GC content",
       main = "GC Bias in Corrected Readcounts", ...)
}

findbin <- function(rbin, sbin){
  rbin$sstart <- rbin$start
  rbin$send   <- rbin$end
  rbin$meanlog2CN <- rbin$logcopy
  
  for (i in seq(dim(sbin)[1])){
    kidx = rbin$chrom == sbin$chrom[i] &
      rbin$start >= sbin$start[i] & 
      rbin$end   <= sbin$end[i]
    rbin$sstart[kidx] <- sbin$start[i]
    rbin$send[kidx]   <- sbin$end[i]
    rbin$meanlog2CN[kidx & !is.na(rbin$logcopy)] <- sbin$meanlog2CN[i]
  }
  
  rbin$CNtype <- NA
  rbin$CNtype[rbin$meanlog2CN<=0 & !is.na(rbin$logcopy)] <- 'HOMD'
  rbin$CNtype[rbin$meanlog2CN>0  & rbin$meanlog2CN<=1 &
                                   !is.na(rbin$logcopy)] <- 'HETD'
  rbin$CNtype[rbin$meanlog2CN>1  & rbin$meanlog2CN<3 &
                !is.na(rbin$logcopy)] <- 'NEUT'
  rbin$CNtype[rbin$meanlog2CN>=3  & rbin$meanlog2CN<4 &
                !is.na(rbin$logcopy)] <- 'GAIN'
  rbin$CNtype[rbin$meanlog2CN>=4  & rbin$meanlog2CN<5 &
                !is.na(rbin$logcopy)] <- 'AMPL'
  rbin$CNtype[rbin$meanlog2CN>=5 & !is.na(rbin$logcopy)] <- 'HLAMP'
  #HOMD Homozygous deletion, ≤ 0 copies
  #HETD Heterozygous deletion, 1 copy
  #NEUT Neutral change, 2 copies
  #GAIN Gain of chromosome, 3 copies
  #AMPL Amplification event, 4 copies
  #HLAMP High level amplification, ≥ 5 copies
  return (rbin)
}

plotCNV <-function(corseg, out, heig=4){
  library(ggplot2)
  library(ggnewscale)
  levec <- as.character(order(as.numeric(unique(corseg$chrom))))
  corseg$chrom <- factor(corseg$chrom, levels=levec)
  g <- ggplot(corseg, aes(x=start, y=logcopy)) + 
      geom_point(size=0.05, aes(color = logcopy) ) +
      scale_colour_gradientn(colours=c("blue", "red")) +
      new_scale_colour() + 
      geom_segment(aes(sstart, meanlog2CN, xend=send, yend=meanlog2CN, color=meanlog2CN)) +
      scale_colour_viridis_c("meamlogcopy", begin=1, end=0.4) 
  #scale_colour_gradientn(colours=c("green", "red"))
  g <- g + labs(x="", y="CNV_log2count profile\n")
  g <- g + facet_grid(SID~chrom, space="free_x", scales="free_x", margins=FALSE)
  g <- g + theme(text= element_text(color = "black",size=11),
                 legend.text=element_text(color = "black",size = 11),
                 panel.background=element_rect(color="black"),
                 axis.text.x = element_text(angle = 90), 
                 #panel.grid.minor = element_blank(), 
                 #panel.grid.major = element_blank(),
                 axis.title.x=element_blank())
  #heig <- max(5, length(unique(corseg$sid)))
  ggsave(out, g, width=28, height = heig )
}

library(DNAcopy)
library(HMMcopy)

args <-commandArgs(T)
wd   <- args[1]
hd   <- args[2]
setwd(wd)
IN <- paste0(hd, '.gz')
IN <- gzfile(IN,'rt')
data <- read.csv(IN, sep='\t')
close(IN)

chroms <- c(as.character(seq(1,22)), c('X', 'Y'))
data  <- data[data$chrom %in% chroms,]
data$chrom <- as.character(data$chrom)
data$chrom[data$chrom =='X'] = '23'
data$chrom[data$chrom =='Y'] = '24'
data <- data[order(as.numeric(data$chrom), data$start),]
SID  <- data$SID[1]

####################GC
cornom1 <- gcloessm(data, x='gc',y='counts')
cornom2 <- gclowessu(data, x='gc',y='counts') # when big data, bias is still exists
write.table(cornom1, paste0(hd, '.cor.nom.txt'), sep='\t', row.names = FALSE, na='', quote=FALSE)

pdf(file = paste0(hd, '.cor.nom.pdf'), width = 8, height = 4)
plotCorr(cornom1, X='gc', yraw='counts', ycor='logcopy')
dev.off()

#########################segmentation
segdata <- cornom1[,c('chrom', 'start', 'end', 'logcopy')]
colnames(segdata) <- c('chrom', 'start', 'end', 'copy')
segdata$chr <- as.factor(segdata$chr)
segcbs <- segmentcbs(segdata, sid=SID) #segmenthmm(segdata)
corseg <- findbin(cornom1, segcbs)
write.table(corseg, paste0(hd, '.cor.seg.txt'), sep='\t', row.names = FALSE, na='', quote=FALSE)
######################plot
# To use for fills, add
plotCNV(corseg, paste0(hd, '.cor.seg.pdf'))




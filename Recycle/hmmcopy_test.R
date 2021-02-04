### R code from vignette source 'HMMcopy.Rnw'

###################################################
### code chunk number 1: HMMcopy.Rnw:92-99
###################################################
options(stringsAsFactors = TRUE)
library(HMMcopy)
rfile <- system.file("extdata", "normal.wig", package = "HMMcopy")
gfile <- system.file("extdata", "gc.wig", package = "HMMcopy")
mfile <- system.file("extdata", "map.wig", package = "HMMcopy")
normal_reads <- wigsToRangedData(rfile, gfile, mfile)
normal_reads[1000:1010, ]
hist(normal_reads$gc, xlim=c(-1,1))
hist(normal_reads$map)

###################################################
### code chunk number 2: HMMcopy.Rnw:109-112
###################################################
normal_copy <- correctReadcount(normal_reads)
normal_copy[1000:1010, ]



###################################################
### code chunk number 3: HMMcopy.Rnw:135-138
###################################################
par(cex.main = 0.7, cex.lab = 0.7, cex.axis = 0.7, mar = c(4, 4, 2, 0.5))
plotBias(normal_copy, pch = 20, cex = 0.5)



###################################################
### code chunk number 4: HMMcopy.Rnw:151-154
###################################################
par(mar = c(4, 4, 2, 0))
plotCorrection(normal_copy, pch = ".")



###################################################
### code chunk number 5: HMMcopy.Rnw:168-174
###################################################
tfile <- system.file("extdata", "tumour.wig", package = "HMMcopy")
tumour_copy <- correctReadcount(wigsToRangedData(tfile, gfile, mfile))

par(mar = c(4, 4, 2, 0))
plotCorrection(tumour_copy, pch = ".")



###################################################
### code chunk number 6: HMMcopy.Rnw:195-197
###################################################
tumour_segments <- HMMsegment(tumour_copy)

kk = HMMsegment(tumour_copy[,c(1,2,3,11)]) 

###################################################
### code chunk number 7: HMMcopy.Rnw:205-214
###################################################
par(mfrow = c(1, 1))
par(cex.main = 0.5, cex.lab = 0.5, cex.axis = 0.5, mar = c(2, 1.5, 0, 0), mgp = c(1, 0.5, 0))
plotSegments(tumour_copy, tumour_segments, pch = ".",
             ylab = "Tumour Copy Number", xlab = "Chromosome Position")

cols <- stateCols() # 6 default state colours
legend("topleft", c("HOMD", "HETD", "NEUT", "GAIN", "AMPL", "HLAMP"),
       fill = cols, horiz = TRUE, bty = "n", cex = 0.5)



###################################################
### code chunk number 8: HMMcopy.Rnw:241-244
###################################################
default_param <- HMMsegment(tumour_copy, getparam = TRUE)
default_param



###################################################
### code chunk number 9: HMMcopy.Rnw:260-265
###################################################
longseg_param <- default_param
longseg_param$e <- 0.999999999999999
longseg_param$strength <- 1e30
longseg_segments <- HMMsegment(tumour_copy, longseg_param, verbose = FALSE)



###################################################
### code chunk number 10: HMMcopy.Rnw:271-277
###################################################
par(cex.main = 0.5, cex.lab = 0.5, cex.axis = 0.5, mar = c(2, 1.5, 0, 0), mgp = c(1, 0.5, 0))
plotSegments(tumour_copy, longseg_segments, pch = ".",
             ylab = "Tumour Copy Number", xlab = "Chromosome Position")
legend("topleft", c("HOMD", "HETD", "NEUT", "GAIN", "AMPL", "HLAMP"),
       fill = cols, horiz = TRUE, bty = "n", cex = 0.5)



###################################################
### code chunk number 11: HMMcopy.Rnw:292-294
###################################################
longseg_segments$mus



###################################################
### code chunk number 12: HMMcopy.Rnw:300-302
###################################################
longseg_param$mu



###################################################
### code chunk number 13: HMMcopy.Rnw:308-318
###################################################
par(cex.main = 0.5, cex.lab = 0.5, cex.axis = 0.5, mar = c(2, 1.5, 0, 0), mgp = c(1, 0.5, 0))
plotSegments(tumour_copy, longseg_segments, pch = ".",
             ylab = "Tumour Copy Number", xlab = "Chromosome Position")
for(i in 1:nrow(longseg_segments$mus)) {
  abline(h = longseg_segments$mus[i ,ncol(longseg_segments$mus)], col = cols[i],
         lwd = 2, lty = 3)
}
abline(v = 7.68e7, lwd = 2, lty = 3)
abline(v = 8.02e7, lwd = 2, lty = 3)



###################################################
### code chunk number 14: HMMcopy.Rnw:334-341
###################################################
newmu_param <- longseg_param
newmu_param$mu <- c(-0.5, -0.4, -0.15, 0.1, 0.4, 0.7)
newmu_segments <- HMMsegment(tumour_copy, newmu_param, verbose = FALSE)
par(cex.main = 0.5, cex.lab = 0.5, cex.axis = 0.5, mar = c(2, 1.5, 0, 0), mgp = c(1, 0.5, 0))
plotSegments(tumour_copy, newmu_segments, pch = ".",
             ylab = "Tumour Copy Number", xlab = "Chromosome Position")



###################################################
### code chunk number 15: HMMcopy.Rnw:354-357
###################################################
newmu_segments$mus
longseg_param$mu



###################################################
### code chunk number 16: HMMcopy.Rnw:366-372
###################################################
par(cex.main = 0.5, cex.lab = 0.5, cex.axis = 0.5, mar = c(2, 1.5, 0, 0), mgp = c(1, 0.5, 0))
newmu_param$m <- newmu_param$mu
realmu_segments <- HMMsegment(tumour_copy, newmu_param, verbose = FALSE)
plotSegments(tumour_copy, realmu_segments, pch = ".",
             ylab = "Tumour Copy Number", xlab = "Chromosome Position")



###################################################
### code chunk number 17: HMMcopy.Rnw:396-400
###################################################
somatic_copy <- tumour_copy
# LOGARITHM IDENTITY: log(a) - log(b) == lob(a / b)
somatic_copy$copy <- tumour_copy$copy - normal_copy$copy



###################################################
### code chunk number 18: HMMcopy.Rnw:405-410
###################################################
somatic_segments <- HMMsegment(somatic_copy, newmu_param, verbose = FALSE)
par(cex.main = 0.5, cex.lab = 0.5, cex.axis = 0.5, mar = c(2, 1.5, 0, 0), mgp = c(1, 0.5, 0))
plotSegments(somatic_copy, somatic_segments, pch = ".",
             ylab = "Tumour Copy Number", xlab = "Chromosome Position")



###################################################
### code chunk number 19: HMMcopy.Rnw:420-422
###################################################
toLatex(sessionInfo())




dnacopy_test <- function(){
  library(DNAcopy)
  data(coriell)
  
  CNA.object <- CNA(cbind(coriell$Coriell.05296),
                    coriell$Chromosome,coriell$Position,
                    data.type="logratio",sampleid="c05296")
  
  smoothed.CNA.object <- smooth.CNA(CNA.object)
  segment.smoothed.CNA.object <- segment(smoothed.CNA.object, verbose=1)
  plot(segment.smoothed.CNA.object, plot.type="w")
  plot(segment.smoothed.CNA.object, plot.type="s") 
  plot(segment.smoothed.CNA.object, plot.type="p")
  
  sdundo.CNA.object <- segment(smoothed.CNA.object, 
                               undo.splits="sdundo", 
                               undo.SD=3,verbose=1)
  plot(sdundo.CNA.object,plot.type="s")
}

gc.correct <- function(coveragA, biaA) {
  NONAIdx  <- which(! is.na(coveragA))
  coverage <- coveragA[NONAIdx]     ## excluse NA Inf value in  coverage
  coveragB <- rep(NA, length(coveragA))
  bias     <-biaA[NONAIdx]
  
  i <- seq(min(bias, na.rm=TRUE), max(bias, na.rm=TRUE), by = 0.001)
  coverage.trend <- loess(coverage ~ bias)
  coverage.model <- loess(predict(coverage.trend, i) ~ i)
  coverage.pred <- predict(coverage.model, bias)
  coverage.corrected <- coverage - coverage.pred + median(coverage)
  coveragB[NONAIdx] <- coverage.corrected
  return (coveragB)
}

lowess.gc <- function(jtkx, jtky) {
  jtklow <- lowess(jtkx, log(jtky), f=0.05)
  jtkz <- approx(jtklow$x, jtklow$y, jtkx)
  return(exp(log(jtky) - jtkz$y))
}
lowess.gc1 <- function(jtkx, jtky) {
  jtklow <- lowess(jtkx, log(jtky), f=0.05)
  jtkz <- approx(jtklow$x, jtklow$y, jtkx)
  print(jtkz$y)
  print(jtklow$y)
  return(exp(log(jtky) - jtkz$y))
}

lowess.gc(cars$speed, cars$dist)


plot(cars)
points(approx(cars, cars$speed), col = 2, pch = "*")
points(approx(cars$speed, cars$dist, cars$speed), col = 4, pch = "*")


########lowess
require(graphics)
plot(cars, main = "lowess(cars)")
lines(lowess(cars), col = 2)
lines(cars$speed, lowess.gc(cars$speed, cars$dist) )

lines(lowess(cars, f = .2), col = 3)
legend(5, 120, c(paste("f = ", c("2/3", ".2"))), lty = 1, col = 2:3)

########loess
cars.lo <- loess(dist ~ speed, cars)
plot(cars, main = "lowess(cars)")
lines(cars$speed, cars.lo$fitted, col = "dark red")

x1 = data.frame(speed = seq(5, 24, 1))
y1 = predict(cars.lo, x1, se = FALSE)
lines(x1, y1, col = 3)

# to allow extrapolation
cars.lo2 <- loess(dist ~ speed, cars,span =0.1,
                  control = loess.control(surface = "direct"))
lines(cars$speed, cars.lo2$fitted, col = "blue")
#predict(cars.lo2, data.frame(speed = seq(5, 30, 1)), se = TRUE)

cars.lo3 <- lowess(cars$speed, cars$dist, f=0.6667)
lines( cars.lo3$x, cars.lo3$y, col = 'green')

jtkz <- approx(cars.lo3$x, cars.lo3$y, cars$speed)
lines( cars.lo3$x, jtkz$y, col = 'red')
cars$dist - jtkz$y
cars$dist - cars.lo3$y

y2 = gc.correct(cars$dist, cars$speed)
lines( cars.lo3$x, y2, col = 'cyan')


approx_test <-function(){
  
  df<-data.frame(x=c(1,2,3),y=c(2,4,8),colour=c("a","b","c"))
  ggplot(df,aes(x,y,colour=factor(colour)))+geom_line(aes(group=1))
  xgrid<-with(df,seq(min(x),max(x),length=15))
  xgrid
  approx(df$x, df$y,xout=c(NA, 1.1, 2, 2.3, 3, 4))
  kk =  df$x+10
  approx(df$x, df$y, kk)
}

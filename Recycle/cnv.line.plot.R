library(ggplot2)

args=commandArgs(T)
INF=args[1]
OU =args[2]

setwd(OU)
cnv=read.csv(INF, sep='\t', header = TRUE)
sid=length(unique( cnv$SID ))

g <- ggplot(cnv, aes(x=Bins, y=log2_depth, color = log2_depth)) + 
        geom_point(size=0.3) +
        geom_line(linetype=2, color='gray') +
        scale_color_gradient(low="blue", high="red")
g <- g + labs(x="", y="CNV_log2count profile\n")
g <- g + facet_grid(SID~GrpC, space="free_x", scales="free_x", margins=FALSE)
ggsave("./log2_depth.pdf", g, width=27, height = sid)

p <- ggplot(cnv, aes(x=Bins, y=log2_count, color = log2_count)) + 
  geom_point(size=0.3) +
  geom_line(linetype=2, color='gray') +
  scale_color_gradient(low="blue", high="red")
p <- p + labs(x="", y="CNV_log2count profile\n")
p <- p + facet_grid(SID~GrpC, space="free_x", scales="free_x", margins=FALSE)
ggsave("./log2_count.pdf", p, width=27, height = sid)


# g <- g + stat_summary(aes(group = type), geom = "line", fun.y = median, size = 1.5, color="steelblue")
# g <- g + labs(x="", y="Fragmentation profile\n", color="")
#g <- g + facet_grid(cols=vars(SID), switch="x",space="free_x", scales="free_x")
#, labeller=labeller(type=tissue,
#                                                                                            arm=arm.labels))
#g <- g + coord_cartesian(xlim = NULL, ylim=c(-.10,.12), expand = TRUE)
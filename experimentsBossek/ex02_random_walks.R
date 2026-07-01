library(devtools)
library(ggplot2)
library(viridis)
library(tidyverse)

# RANDOM WALKS
# ===
#
# Analyse the footprints of the random walks.
#

plotTraces = function(df, extreme.sols = NULL) {
  df$max.sigma = as.factor(df$max.sigma)
  pl = ggplot(df)
  pl = pl + geom_segment(aes(x = p1, y = p2, xend = c1, yend = c2, colour = as.factor(repl)), arrow = arrow(length = unit(0.15, "cm")))
  if (length(unique(df$class)) > 1L) {
    pl = pl + facet_wrap(class ~ algorithm, ncol = 10, scales = "free")
  } else {
    pl = pl + facet_wrap(. ~ algorithm, ncol = 10, scales = "free")
  }
  pl = pl + labs(
    colour = "Random walk run",
    x = expression(c[1]),
    y = expression(c[2])
  )
  pl = pl + theme_minimal()
  pl = pl + theme(legend.position = "top")
  pl = pl + viridis::scale_colour_viridis(discrete = TRUE, end = 0.85)
  pl = pl + scale_color_brewer(palette = "Dark2")
  if (!is.null(extreme.sols)) {
    extreme.sols = lapply(unique(df$max.sigma), function(ms) {
      tmp = extreme.sols
      tmp$max.sigma = ms
      tmp
    })
    extreme.sols = do.call(rbind, extreme.sols)
    pl = pl + geom_point(data = extreme.sols, aes(x = c1, y = c2))
  }

  return(pl)
}

res = read.table("results/random_walks.csv", sep = " ", header = TRUE, stringsAsFactors = FALSE)
#res$doscal = ifelse(res$scalarize, "on", "off")
res$mut = sprintf("%s (%i)", res$mut, res$max.sigma)
res$class = res$instance
res$algorithm = res$mut
res = filter(res, repl <= 3)

pl = plotTraces(res)#, extreme.sols = extrsols))
ggsave("figures/random_walks.pdf", plot = pl, width = 60, height = 50, limitsize = FALSE)

stop()
# PAPER PLOTS:
# ===
# now we show an excerpt of the other operators, i.e., only a few instances and
# other parameterizations
ress = filter(res,
  class == "C1",
  mut %in% c("SG (25)", "SGS (25)", "USG (25)", "USGS (25)"))

ress = ecr::implode(ress, cols = c("class", "mut"), by = " - ", col.name = "algorithm2", keep = TRUE)
ress2 = filter(ress, class == "C1")
ress2$algorithm2 = factor(ress2$algorithm2, levels = c("C1 - SG (25)", "C1 - SGS (25)", "C1 - USG (25)", "C1 - USGS (25)"), ordered = TRUE)

pl = plotTraces(ress2)
pl = pl + facet_grid(. ~ algorithm2)
ggsave("figures/random_walks/random_walks_on_c1.pdf", plot = pl, width = 10, height = 3.4)

ress = filter(res,
  class == "C2",
  mut %in% c("SG (25)", "SGS (25)", "USG (25)", "USGS (25)"))

ress = ecr::implode(ress, cols = c("class", "mut"), by = " - ", col.name = "algorithm2", keep = TRUE)
ress2 = filter(ress, class == "C2")
ress2$algorithm2 = factor(ress2$algorithm2, levels = c("C2 - SG (25)", "C2 - SGS (25)", "C2 - USG (25)", "C2 - USGS (25)"), ordered = TRUE)

pl = plotTraces(ress2)
pl = pl + facet_grid(. ~ algorithm2)
ggsave("figures/random_walks/random_walks_on_c2.pdf", plot = pl, width = 10, height = 3.4)

ress = filter(res,
  class == "C3",
  mut %in% c("SG (25)", "SGS (25)", "USG (25)", "USGS (25)"))

ress = ecr::implode(ress, cols = c("class", "mut"), by = " - ", col.name = "algorithm2", keep = TRUE)
ress2 = filter(ress, class == "C3")
ress2$algorithm2 = factor(ress2$algorithm2, levels = c("C3 - SG (25)", "C3 - SGS (25)", "C3 - USG (25)", "C3 - USGS (25)"), ordered = TRUE)

pl = plotTraces(ress2)
pl = pl + facet_grid(. ~ algorithm2)
ggsave("figures/random_walks/random_walks_on_c3.pdf", plot = pl, width = 10, height = 3.4)

ress = filter(res,
  class == "C4",
  mut %in% c("SG (25)", "SGS (25)", "USG (25)", "USGS (25)"))

ress = ecr::implode(ress, cols = c("class", "mut"), by = " - ", col.name = "algorithm2", keep = TRUE)
ress2 = filter(ress, class == "C4")
ress2$algorithm2 = factor(ress2$algorithm2, levels = c("C4 - SG (25)", "C4 - SGS (25)", "C4 - USG (25)", "C4 - USGS (25)"), ordered = TRUE)

pl = plotTraces(ress2)
pl = pl + facet_grid(. ~ algorithm2)
ggsave("figures/random_walks/random_walks_on_c4.pdf", plot = pl, width = 10, height = 3.4)


# SUPLEMENTARY MATERIAL PLOTS:
# ===

ress = filter(res,
  #class %in% c("C1", "C4"),
  max.sigma %in% c("7", "25"))

ress = ecr::implode(ress, cols = c("class", "mut"), by = " - ", col.name = "algorithm2", keep = TRUE)

for (cl in unique(ress$class)) {
  pl = plotTraces(filter(ress, class == cl))
  pl = pl + facet_grid(. ~ algorithm2)#, scales = "free", nrow = 4)
  fn = sprintf("figures/random_walks/random_walks_on_%s_of_SG_SGS_USG_USGS.pdf", cl)
  ggsave(fn, plot = pl, width = 16, height = 3.9)
}


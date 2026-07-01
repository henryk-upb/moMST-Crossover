library(dplyr)
library(ecr)

# CALCULATE EMOA INDICATORS
# ===
#
# This script imports the results of the benchmark study and calcualtes
# some EMOA performance indicators via the ecr package and again stores
# the results.
#

res.all = readr::read_delim("outdata/ex04_pf.csv", delim = ",")#, col_types = "ddcciic")
colnames(res.all) = c("c1", "c2", "jobid", "prob", "algorithm", "repl", "class", "no")

res.all = ecr::normalize(res.all, obj.cols = c("c1", "c2"), offset = 1)

## PERFORMANCE INDICATORS
## =======

unary.inds = list(
  HV = list(fun = ecr::emoaIndHV),
  EPS = list(fun = ecr::emoaIndEps),
  IGD = list(fun = ecr::emoaIndIGD)
)

set.seed(1)
parallelMap::parallelStartMulticore(cpus = 7L)
inds = ecr::computeIndicators(res.all, obj.cols = c("c1", "c2"), unary.inds = unary.inds)
parallelMap::parallelStop()

inds.unary = as_tibble(inds$unary)
meta = filter(unique(res.all[, c("algorithm", "prob", "class", "repl", "no", "jobid")]))

inds.unary = left_join(inds.unary, meta, by = c("prob", "algorithm", "repl"))
colnames(inds.unary) = c("mutator", "instance", "repl", "HV", "EPS", "IGD", "class", "no", "jobid")
write.table(inds.unary, file = "outdata/ex04_indicators.csv", row.names = FALSE, col.names = TRUE, quote = TRUE, sep=",")

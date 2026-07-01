library(dplyr)
library(kableExtra)

devtools::load_all("~/code/libs/r/tblutils")

res.all = readr::read_delim("outdata/ex04_indicators.csv", delim = ",", col_types = "ccidddcii")

# get all instances
algo.names = c("USGS-F", "USGS-PRE", "SGS", "IF-G")
algo.colors = colors = c("violet", "brown", "teal", "purple")

# Filter relevant data:
tbl = res.all %>%
  filter(mutator %in% algo.names) %>%
  dplyr::select(class, no, mutator, HV)

tbl$mutator2 = factor(tbl$mutator, levels = algo.names, ordered = TRUE)
tbl$class = gsub("CLASS", "C", tbl$class)
tbl = arrange(tbl, mutator2)

res = to_result_table(tbl,
  split.cols = c("class", "no"),
  widen.col = c("mutator2"),
  measure.cols = c("HV"),
  test.alternative = c(HV = "less"),
  testresult.formatter.args = list(positive.only = TRUE, interval = FALSE, colors = algo.colors),
  stats.formatter.args = list(sd = list(digits = 2)),
  highlighter.args = list(order.fun = "min", bg.color = "gray", bg.saturation.max = 0, digits = 2L))

tbl.HV = res[[1L]]
tbl.HV$mean.HV = sprintf("%s $\\pm$ %.2f", tbl.HV$mean.HV, tbl.HV$sd.HV)
tbl.HV$mean.HV2 = sprintf("%s $\\pm$ %.2f", tbl.HV$mean.HV2, tbl.HV$sd.HV2)
tbl.HV$mean.HV3 = sprintf("%s $\\pm$ %.2f", tbl.HV$mean.HV3, tbl.HV$sd.HV3)
tbl.HV$mean.HV4 = sprintf("%s $\\pm$ %.2f", tbl.HV$mean.HV4, tbl.HV$sd.HV4)
tbl.HV$sd.HV = tbl.HV$sd.HV2 = tbl.HV$sd.HV3 = tbl.HV$sd.HV4 = NULL

ktbl = to_latex(
  tbl.HV,
  reps = 1L,
  param.col.names = c("CLASS", "$I$"),
  measure.col.names = c("\\textbf{mean $\\pm$ sd}", "\\textbf{stat}"),
  algo.names = algo.names,
  algo.colors = algo.colors,
  caption = "Mean values and results of Wilcoxon-Mann-Whitney tests at significance level $\\alpha=0.01$ (\\textbf{stat}) with respect to the HV-indicator (the closer to zero, the better). The \\textbf{stat}-column is to be read as follows: a value $X^{+}$ indicates that the indicator for the column algorithm (note that algorithms are numbered and color-encoded in the second row) is significantly lower than the one of algorithm $X$. Lowest indicator values are highlighted in \\textbf{bold-face}.") %>%
  kable_styling() %>%
  #row_spec(row = c(10, 20, 30), extra_latex_after = "\\cmidrule{2-14}") %>%
  collapse_rows(columns = 1:2, latex_hline = "major", valign = "middle")
  #add_header_above(c(" ", " ", "HV-indicator" = 12), bold = TRUE, escape = FALSE)
#preview(ktbl)
cat(ktbl, file = "tables/indicators.tex")


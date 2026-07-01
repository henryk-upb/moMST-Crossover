library(ggplot2)
#library(tidyverse)
library(latex2exp)
library(reshape2)

res = readr::read_delim("results/ex01_usgs.csv", delim = " ")
print(res)

res = dplyr::filter(res, p == 1, n %in% c(100, 400, 700, 1000))#, sigma.fun != "log")
res$sigma.fun = factor(res$sigma.fun, levels = c("log", "sqrt", "logsquared", "half"), ordered = TRUE)

# transform to long format
res = reshape2::melt(res, id.vars = c("n", "p", "sigma.fun", "sigma", "repl"), variable.name = "Operator", value.name = "runtime")
#res$runtime = res$runtime / (res$n^2 * log(res$n, base = 2))

# nice LaTeX facet labels
levels(res$sigma.fun) = c(log = TeX("$s = \\sigma = \\log(n)$"), sqrt = TeX("$s = \\sigma = \\sqrt{n}$"), logsquared = TeX("$s = \\sigma = \\log^2(n)$"), half = TeX("$s = \\sigma = n/2$"))

# Box-plots: USGS vs. USGS-F vs. USGS-Pre
# ===

g = ggplot(res, aes(x = as.factor(n), y = runtime, color = Operator))
g = g + geom_boxplot()
g = g + facet_wrap(. ~ sigma.fun, labeller = label_parsed, nrow = 1)
g = g + theme_minimal()
g = g + scale_color_brewer(palette = "Dark2")
g = g + theme(legend.position = "top", axis.text.x = element_text(vjust = 1, angle = 45))
g = g + labs(x = "n", y = "Runtime [in seconds]", color = "")
g

ggsave("figures/runtimes/usgs.pdf", width = 6, height = 3, device = cairo_pdf)

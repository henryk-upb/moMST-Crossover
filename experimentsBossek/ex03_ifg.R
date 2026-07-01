library(ggplot2)
library(dplyr)
library(latex2exp)
library(reshape2)

res = readr::read_delim("results/ex03_ifg.csv", delim = " ")
print(res)

# Box-plots: IF-G for various n and sigma
# ===

g = ggplot(res, aes(x = as.factor(sigmafrac), y = runtime, color = as.factor(n)))
g = g + geom_boxplot()
g = g + theme_minimal()
g = g + scale_color_brewer(palette = "Dark2")
g = g + theme(legend.position = "top", axis.text.x = element_text(vjust = 1, angle = 45))
g = g + labs(x = TeX("n"), y = "Runtime [in seconds]", color = "n")
g

ggsave("figures/runtimes/ifg.pdf", width = 6, height = 3, device = cairo_pdf)

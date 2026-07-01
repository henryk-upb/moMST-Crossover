library(ggplot2)
library(dplyr)
library(latex2exp)
library(reshape2)

res = readr::read_delim("results/ex02_usgs_bounds.csv", delim = " ")
res$m = res$n * (res$n - 1) / 2
print(res)

res$sigma.fun = factor(res$sigma.fun, levels = c("log", "sqrt", "logsquared", "half"), ordered = TRUE)
res = filter(res, sigma.fun != "half")

# transform to long format
# res = reshape2::melt(res, id.vars = c("n", "p", "sigma.fun", "sigma", "repl"), variable.name = "Operator", value.name = "runtime")

# nice LaTeX facet labels
levels(res$sigma.fun) = c(log = TeX("$s = \\sigma = \\log(n)$"), sqrt = TeX("$s = \\sigma = \\sqrt{n}$"), logsquared = TeX("$s = \\sigma = \\log^2(n)$"), half = TeX("$s = \\sigma = n/2$"))


# Error-plots: USGS vs. USGS-F (with edge filtering)
# ===

res2 = filter(res, n %in% seq(200, 1000, by = 200))
res2 = res
g = ggplot(res2, aes(x = as.factor(n), y = value / m, color = as.factor(sigma.fun)))
#g = g + geom_pointrange(aes(ymin = LB, ymax = UB, color = NULL), width = 2, color = "black")
#g = g + geom_errorbar(aes(ymin = LB, ymax = UB, color = NULL), width = 2, color = "black")
g = g + geom_boxplot(show.legend = FALSE)
g = g + geom_point(mapping = aes(y = UB/m), color = "gray")
g = g + geom_point(mapping = aes(y = LB/m), color = "gray")
#g = g + geom_jitter(alpha = 0.3)
g = g + theme_minimal()
g = g + scale_color_brewer(palette = "Dark2")
g = g + facet_wrap(. ~ sigma.fun, labeller = label_parsed)
g = g + labs(x = "n", y = "Fraction of relevant edges")
g = g + ylim(c(0, 1))
g = g + theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(g)

ggsave("figures/runtimes/usgs_bounds.pdf", width = 7, height = 3, device = cairo_pdf)


stop()



g = ggplot(res, aes(x = as.factor(n), y = value / UB, group = n))
g = g + geom_violin()
g = g + geom_jitter(alpha = 0.3)
g = g + theme_minimal()
g = g + facet_wrap(. ~ sigma.fun, labeller = label_parsed)
g = g + labs(x = "n", y = "Fraction of relevant edges")
g

ggsave("figures/runtimes/usgs_bounds.pdf", width = 10, height = 3.5, device = cairo_pdf)


# g = ggplot(res, aes(x = as.factor(n), y = value / UB, color = sigma.fun))
# g = g + geom_violin()
# g = g + geom_jitter()
# g = g + theme_minimal()
# #g = g + facet_wrap(. ~ sigma.fun, labeller = label_parsed)
# g = g + labs(x = "n", y = "Absolute no. of relevant edges")
# g

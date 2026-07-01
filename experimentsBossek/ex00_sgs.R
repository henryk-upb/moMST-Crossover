library(ggplot2)
library(dplyr)
library(latex2exp)
library(reshape2)

# rt.files = list.files("results/runtime", pattern = "runtime", full.names = TRUE, all.files = TRUE)
# print(rt.files)
# rt.files

# import = function(x) {
#     print(x)
#     spl = strsplit(x, split = "_", fixed = TRUE)[[1]]
#     print(spl)

#     data = readr::read_delim(x, delim = " ")
#     data$interpreter = spl[1L]
#     data$chash = spl[2L]
#     return (data)
# }

# res = do.call(rbind, lapply(rt.files, import))

# print(res)

# res$sigma.fun = factor(res$sigma.fun, levels = c("log", "sqrt", "half"), ordered = TRUE)

# # transform to long format
# res = reshape2::melt(res, id.vars = c("n", "sigma.fun", "sigma", "repl", "chash", "interpreter"), variable.name = "Operator", value.name = "runtime")
# # nice LaTeX facet labels
# levels(res$sigma.fun) = c(log = TeX("$s = \\sigma = \\log(n)$"), sqrt = TeX("$s = \\sigma = \\sqrt{n}$"), half = TeX("$s = \\sigma = n/2$"))

# g = ggplot(res, aes(x = as.factor(n), y = runtime, color = interaction(Operator, chash)))
# g = g + geom_boxplot()
# g = g + facet_wrap(. ~ sigma.fun, labeller=label_parsed)
# g = g + theme_minimal()
# g = g + scale_color_brewer(palette = "Dark2")
# g = g + theme(legend.position = "top")
# g = g + labs(x = "n", y = "Runtime [in seconds]", color = "")
# g

# ggsave("figures/runtime/runtimes_opt_process.pdf", width = 12, height = 5, device = cairo_pdf)


res = readr::read_delim("results/ex00_sgs.csv", delim = " ")
res = filter(res, n %in% c(100, 400, 700, 1000))
res$sigma.fun = factor(res$sigma.fun, levels = c("log", "sqrt", "logsquared", "half"), ordered = TRUE)

# transform to long format
res = reshape2::melt(res, id.vars = c("n", "sigma.fun", "sigma", "repl"), variable.name = "Operator", value.name = "runtime")
# nice LaTeX facet labels
res = filter(res, Operator != "SGSI")
levels(res$sigma.fun) = c(log = TeX("$s = \\sigma = \\log(n)$"), sqrt = TeX("$s = \\sigma = \\sqrt{n}$"), logsquared = TeX("$s = \\sigma = \\log^2(n)$"), half = TeX("$s = \\sigma = n/2$"))

g = ggplot(res, aes(x = as.factor(n), y = runtime, color = Operator))
g = g + geom_boxplot()
g = g + facet_wrap(. ~ sigma.fun, labeller = label_parsed, scales = "free", nrow = 1)
g = g + theme_minimal()
g = g + scale_color_brewer(palette = "Dark2")
g = g + theme(legend.position = "top")
g = g + labs(x = "n", y = "Runtime [in seconds]", color = "")
g = g + theme(legend.position = "top", axis.text.x = element_text(vjust = 1, angle = 45))
g

ggsave("figures/runtimes/sgs.pdf", width = 6, height = 3, device = cairo_pdf)

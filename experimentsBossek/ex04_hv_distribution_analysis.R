library(dplyr)
library(ggplot2)

# ==============================================================================
# HV-INDICATOR DISTRIBUTION ANALYSIS (single instance)
# ==============================================================================
#
# Implements the analysis your supervisor suggested:
#
#   For ONE instance, K algorithms and R independent repetitions each, you get
#   R x K Pareto-front approximations. Each front is reduced to a single HV
#   value (already done by ex04_indicators.R via ecr::emoaIndHV), which leaves
#   you with K empirical distributions of R indicator values. These are
#   analysed (a) visually - boxplot, violin plot, histogram, ECDF - and
#   (b) statistically - an omnibus test + pairwise significance tests for
#   deviations in mean/median.
#
# Input: outdata/ex04_indicators.csv (produced by ex04_indicators.R)
#        columns: mutator, instance, repl, HV, EPS, IGD, class, no, jobid
#
# NOTE on direction: as in ex04_indicator_tables.R (test.alternative = "less"),
# lower HV is BETTER here, because ecr::normalize()+emoaIndHV is applied to a
# minimisation front. Keep that in mind when reading "better"/"worse" below.
#
# The single-instance analysis is wrapped in functions (analyse_instance,
# plot_instance) so that "extending to every instance" (mentioned as
# straightforward in the suggestion) is just a loop / facet_wrap(~ instance) -
# see the bottom of this script.
# ==============================================================================

indicators = readr::read_delim("outdata/ex04_indicators.csv", delim = ",",
                                col_types = "ccidddcii")

# algorithms to compare (K): drop IF-L, as done elsewhere in this repo
algo.names = c("USGS-F", "USGS-PRE", "SGS", "IF-G")

# ------------------------------------------------------------------------------
# 1) Pick ONE instance and reduce to the R x K table of HV values
# ------------------------------------------------------------------------------

get_instance_data = function(indicators, class.name, instance.no) {
  df = indicators %>%
    filter(class == class.name, no == instance.no, mutator %in% algo.names) %>%
    transmute(algorithm = factor(mutator, levels = algo.names), repl, HV)

  # sanity check: exactly R rows per algorithm (R x K "matrix" in long format)
  stopifnot(length(unique(table(df$algorithm))) == 1)
  df
}

sel.class = "CLASS1"
sel.no    = 1L
df = get_instance_data(indicators, sel.class, sel.no)
sel.label = paste0(sel.class, "_", sel.no)

# ------------------------------------------------------------------------------
# 2) VISUALISE the K empirical distributions
# ------------------------------------------------------------------------------

plot_instance = function(df, label) {
  pal = "Dark2"  # consistent with the rest of the repo (scale_*_brewer)

  p_box = ggplot(df, aes(x = algorithm, y = HV, fill = algorithm)) +
    geom_boxplot(outlier.alpha = 0.5) +
    scale_fill_brewer(palette = pal) +
    theme_minimal() + theme(legend.position = "none") +
    labs(x = "Algorithm", y = "HV", title = paste("Boxplot -", label))

  p_violin = ggplot(df, aes(x = algorithm, y = HV, fill = algorithm)) +
    geom_violin(trim = FALSE, alpha = 0.7) +
    geom_jitter(width = 0.08, alpha = 0.4, size = 0.8) +
    scale_fill_brewer(palette = pal) +
    theme_minimal() + theme(legend.position = "none") +
    labs(x = "Algorithm", y = "HV", title = paste("Violin plot -", label))

  p_hist = ggplot(df, aes(x = HV, fill = algorithm)) +
    geom_histogram(bins = 15, alpha = 0.8) +
    facet_wrap(~ algorithm, scales = "free_y", ncol = 1) +
    scale_fill_brewer(palette = pal) +
    theme_minimal() + theme(legend.position = "none") +
    labs(x = "HV", y = "Count", title = paste("Histograms -", label))

  p_ecdf = ggplot(df, aes(x = HV, color = algorithm)) +
    stat_ecdf(geom = "step", linewidth = 0.9) +
    scale_color_brewer(palette = pal) +
    theme_minimal() +
    labs(x = "HV", y = "ECDF", color = "Algorithm", title = paste("ECDF -", label))

  list(box = p_box, violin = p_violin, hist = p_hist, ecdf = p_ecdf)
}

plots = plot_instance(df, sel.label)

out.dir = file.path("figures", "hv_distribution")
dir.create(out.dir, showWarnings = FALSE, recursive = TRUE)
ggsave(file.path(out.dir, paste0(sel.label, "_boxplot.pdf")),   plots$box,    width = 5, height = 4)
ggsave(file.path(out.dir, paste0(sel.label, "_violin.pdf")),    plots$violin, width = 5, height = 4)
ggsave(file.path(out.dir, paste0(sel.label, "_histogram.pdf")), plots$hist,   width = 5, height = 6)
ggsave(file.path(out.dir, paste0(sel.label, "_ecdf.pdf")),      plots$ecdf,   width = 5, height = 4)

# quick on-screen combined view while working interactively (optional)
if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork)
  print((plots$box + plots$violin) / (plots$hist + plots$ecdf))
}

# ------------------------------------------------------------------------------
# 3) TEST for significant deviations in mean/median
# ------------------------------------------------------------------------------

analyse_instance = function(df) {
  # 3a) omnibus test across all K groups (non-parametric one-way ANOVA analogue)
  kw = kruskal.test(HV ~ algorithm, data = df)

  # 3b) pairwise post-hoc: Wilcoxon-Mann-Whitney with Holm correction
  #     (same test family already used for the LaTeX tables in
  #     ex04_indicator_tables.R, just applied directly/exploratively here)
  pw = pairwise.wilcox.test(df$HV, df$algorithm, p.adjust.method = "holm", exact = FALSE)

  # 3c) effect size per pair: rank-biserial correlation (= Cliff's delta)
  #     r = 1 - 2U / (n1 * n2); |r| < .147 negligible, < .33 small, < .474 medium, else large
  algos = levels(df$algorithm)
  pairs = combn(algos, 2, simplify = FALSE)
  effect.sizes = do.call(rbind, lapply(pairs, function(p) {
    x = df$HV[df$algorithm == p[1]]
    y = df$HV[df$algorithm == p[2]]
    wt = wilcox.test(x, y, exact = FALSE)
    u = as.numeric(wt$statistic)
    r = 1 - (2 * u) / (length(x) * length(y))
    data.frame(algo1 = p[1], algo2 = p[2], p.value = wt$p.value, rank_biserial_r = r)
  }))
  effect.sizes$p.adj.holm = p.adjust(effect.sizes$p.value, method = "holm")

  # 3d) descriptive summary
  summary.tbl = df %>%
    group_by(algorithm) %>%
    summarise(n = n(), mean = mean(HV), sd = sd(HV), median = median(HV), IQR = IQR(HV)) %>%
    arrange(mean)  # lower HV = better in this repo's convention

  list(kruskal = kw, pairwise = pw, effect_sizes = effect.sizes, summary = summary.tbl)
}

res = analyse_instance(df)
print(res$summary)
print(res$kruskal)
print(res$pairwise)
print(res$effect_sizes)

# ==============================================================================
# 4) EXTENSION TO EVERY INSTANCE ("straightforward", as noted in the suggestion)
# ==============================================================================
# a) one figure with small multiples across all instances:

p_all = ggplot(indicators %>% filter(mutator %in% algo.names),
                aes(x = factor(mutator, levels = algo.names), y = HV, fill = mutator)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ instance, scales = "free_y") +
  scale_fill_brewer(palette = "Dark2") +
  theme_minimal() +
  theme(legend.position = "top", axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
  labs(x = "Algorithm", y = "HV", fill = "Algorithm")
ggsave(file.path(out.dir, "all_instances_boxplot.pdf"), p_all, width = 12, height = 10)

# b) same statistical test, applied per instance, results collected in one table
#    (this is essentially what ex04_indicator_tables.R already tabulates into LaTeX;
#    here it is just the raw per-instance result list for further inspection)
all.results = lapply(unique(indicators$instance), function(inst) {
  df.i = indicators %>%
    filter(instance == inst, mutator %in% algo.names) %>%
    transmute(algorithm = factor(mutator, levels = algo.names), repl, HV)
  if (nrow(df.i) == 0) return(NULL)
  kw.i = kruskal.test(HV ~ algorithm, data = df.i)
  data.frame(instance = inst, kruskal.p = kw.i$p.value)
})
all.results = do.call(rbind, all.results)
all.results$kruskal.p.adj = p.adjust(all.results$kruskal.p, method = "holm")
print(all.results)

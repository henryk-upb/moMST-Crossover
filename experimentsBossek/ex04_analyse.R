library(ecr)
library(dplyr)
library(readr)
library(ggplot2)
library(latex2exp)

# ToDo's
# ===
# * unify delimiter in design files and outfiles
# * For each jobid store files:
#   - jobid/final.csv: final PF approximation
#   - jobid/history.csv: PF-trajectory for a subset of generations
#.  - jobid/time.csv: CPU/runtime in seconds
# Features:
# * import functions should receive a vector of job.ids (idea: work on
#   setup data frame, filter for interesting stuff and next import the results selectively)

import_trajectories = function(out.path, df.setup) {
    df.setup$jobid = as.integer(df.setup$jobid)

    jobids = as.integer(unique(df.setup$jobid))
    outfiles = file.path(out.path, jobids, "trajectory.csv")
    #outfiles = list.files(out.path, recursive = TRUE, pattern = "trajectory.csv$", full.names = TRUE)
    df.res = lapply(outfiles, function(outfile) {
        jobres = readr::read_delim(outfile, delim = ",")
        # maxgen = max(jobres$gen)
        jobid = as.integer(gsub(".csv", "", basename(dirname(outfile)), fixed = TRUE))
        jobres$jobid = jobid
        return(jobres)
    })
    df.res = do.call(rbind, df.res)
    df.res = dplyr::left_join(df.res, df.setup, by = "jobid")
    return(df.res)
}

import_approximations = function(out.path, setupfile.path) {
    df.setup = readr::read_delim(setupfile.path, delim = ",")
    print(df.setup)
    df.setup$jobid = as.integer(df.setup$jobid)

    outfiles = list.files(out.path, recursive = TRUE, pattern = "pf.csv$", full.names = TRUE)
    #print(outfiles)
    df.res = lapply(outfiles, function(outfile) {
        jobres = readr::read_delim(outfile, delim = ",")
        #jobres[ecr::which.nondominated(t(as.matrix(jobres))), , drop = FALSE]
        jobid = as.integer(gsub(".csv", "", basename(dirname(outfile)), fixed = TRUE))
        jobres$jobid = jobid

        #print(jobres)
        #stop()
        return(jobres)
    })
    df.res = do.call(rbind, df.res)
    df.res = dplyr::left_join(df.res, df.setup, by = "jobid")
    return(df.res)
}

import_runtimes = function(out.path, setupfile.path) {
    df.setup = readr::read_delim(setupfile.path, delim = ",")
    df.setup$jobid = as.integer(df.setup$jobid)

    outfiles = list.files(out.path, recursive = TRUE, pattern = "runtime.csv$", full.names = TRUE)
    df.res = lapply(outfiles, function(outfile) {
        data.frame(runtime = as.numeric(paste(readLines(outfile), collapse = " ")), jobid = as.integer(basename(dirname(outfile))))
    })
    df.res = do.call(rbind, df.res)
    df.res = dplyr::left_join(df.res, df.setup, by = "jobid")
    return(df.res)
}

plot_trajectories = function(df.traj) {
    g = ggplot()
    g = g + geom_point(data = df.traj, mapping = aes(x = c1, y = c2, color = as.factor(gen), shape = as.factor(gen)))
    g = g + theme_minimal()
    g = g + facet_wrap(. ~ instance, scales = "free")
    g = g + scale_color_brewer(palette = "Dark2")
    g = g + labs(color = "Mutator", shape = "Mutator")
    return(g)
}

out.path = "results/approximations/ex04/"
setupfile.path = "parameters/ex04.csv"

# runtimes
df.rt = import_runtimes(out.path, setupfile.path)
df.rt.aggr = df.rt %>%
    group_by(mutator) %>%
    summarise(mean = mean(runtime), sd = sd(runtime)) %>%
    ungroup() %>%
    filter(mutator != "IF-L")

t(as.matrix(df.rt.aggr))

stop()

df.pf = import_approximations(out.path, setupfile.path)
df.pf$instance = basename(df.pf$instance)
df.pf$L = df.pf$seed = NULL
df.pf$class = unname(sapply(df.pf$instance, function(x) {
    strsplit(x, split = "_", fixed = TRUE)[[1L]][1L]
}))
df.pf$no = as.integer(unname(sapply(df.pf$instance, function(x) {
    s = strsplit(x, split = "_", fixed = TRUE)[[1L]]
    s = s[length(s)]
    return(gsub(".graph", "", s))
})))
#df.times = import_times(out.path, setupfile.path)
write.table(df.pf, file = "outdata/ex04_pf.csv", sep = ",", row.names = FALSE)

# PARETO-FRONT APPROXIMATIONS
# ===

df.sel = filter(df.pf, no == 1, repl == 1, mutator != "IF-L")
g = ggplot()
g = g + geom_point(data = df.sel, mapping = aes(x = c1, y = c2, color = as.factor(mutator), shape = as.factor(mutator)), alpha = 0.5)
g = g + theme_minimal()
g = g + facet_wrap(. ~ as.factor(class), scales = "free", nrow = 2)
g = g + scale_color_brewer(palette = "Dark2")
g = g + labs(color = "Mutator", shape = "Mutator")
g = g + theme(legend.position = "top", axis.ticks.x = element_blank(), axis.ticks.y = element_blank(), axis.text.x = element_blank(), axis.text.y = element_blank())
g = g + labs(x = TeX("$c_1$"), y = TeX("$c_2$"))
print(g)
ggsave(pl = g, file = "figures/approximations/scatter.pdf", width = 6, height = 6.4)
stop()

df.setup = readr::read_delim(setupfile.path, delim = ",")

df.sel = filter(df.setup, jobid %in% c(1710, 1715))

df.traj = import_trajectories(out.path, df.sel)

g = plot_trajectories(filter(df.traj, gen %in% seq(400, 500, by = 20))) + facet_wrap(. ~ mutator)

# g = ggplot()
#     g = g + geom_point(data = filter(df.traj, gen == 400), mapping = aes(x = c1, y = c2, color = mutator))#, text = gen))
#     g = g + theme_minimal()
# #    g = g + facet_wrap(. ~ mutator, scales = "free")
# #    g = g + scale_color_brewer(palette = "Dark2")
#     g = g + labs(color = "Mutator", shape = "Mutator")
#     print(g)


#ggsave(pl = g, file = file.path("figures", "trajectory.pdf"), device = cairo_pdf, width = 12, height = 5)

#print(g)


#stop()

#df.traj = import_trajectories(out.path, setupfile.path)


print(df.pf)


stop()


# Why do I need to do this!?!
for (x in unique(df.pf$instance)) {

    df.sel = filter(df.pf, instance == x)#repl %in% 1:10)#, mutator %in% c("USGS-F", "USGS-Pre"))

    g = ggplot()
    g = g + geom_point(data = df.sel, mapping = aes(x = c1, y = c2, color = as.factor(mutator), shape = as.factor(mutator)))
    g = g + theme_minimal()
    g = g + facet_wrap(. ~ as.factor(repl), scales = "free")
    g = g + scale_color_brewer(palette = "Dark2")
    g = g + labs(color = "Mutator", shape = "Mutator")
    g = g + ggtitle(x)
    print(g)
    BBmisc::pause()
    #ggsave(pl = g, file = paste0("figures/scatter/", x, ".pdf"), device = cairo_pdf, width = 20, height = 20)
}



# TRAJECTORIES
# ===

# Why do I need to do this!?!
df.traj$gen = as.integer(df.traj$gen)

final.gen = max(df.traj$gen)

df.final = filter(df.traj, gen == final.gen)

print(unique(df.final$instance))

sel.instances = sample(unique(df.final$instance), size = 1)

df = filter(df.final, instance %in% sel.instances)

g = ggplot()
g = g + geom_point(data = df, mapping = aes(x = c1, y = c2, color = as.factor(mutator), shape = as.factor(mutator)))
g = g + theme_minimal()
g = g + facet_wrap(. ~ instance, scales = "free")
g = g + scale_color_brewer(palette = "Dark2")
g = g + labs(color = "Mutator", shape = "Mutator")
print(g)
ggsave(pl = g, file = "yaaay.pdf", device = cairo_pdf, width = 10, height = 10)

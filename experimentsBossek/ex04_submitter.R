library(batchtools)
library(data.table)

file.dir = "bt-experiments"
out.dir = "data/raw"

load_study = function(setup.file) {
  tbl = readr::read_delim(setup.file, delim = " ")
  tbl = tbl[, "jobid", drop = FALSE]
  tbl$study = gsub(".csv", "", basename(setup.file))
  return(tbl)
}

if (!dir.exists(out.dir))
  dir.create(out.dir, recursive = TRUE)

unlink(file.dir, recursive = TRUE)
reg = batchtools::makeExperimentRegistry(file.dir = file.dir, conf.file = "batchtools_config.R")
#reg$cluster.functions = batchtools::makeClusterFunctionsMulticore(ncpus = parallel::detectCores())

batchtools::addProblem("DUMMY")
batchtools::addAlgorithm("RUNNER", fun = function(job, ...) {
  args = list(...)
  exp.runner.file = sprintf("%s_runner.py", args$study)
  out.file = sprintf("results/raw/%s/%i.csv", args$study, args$jobid)
  if (!dir.exists(dirname(out.file)))
    dir.create(dirname(out.file), showWarnings = FALSE)
  args = c(exp.runner.file, args$jobid, ">", out.file)
  args = BBmisc::collapse(args, sep = " ")
  system2("python3.12", args)
})

# Files with experimental designs
setup.files = "study_data/study04.csv"
design = load_study(setup.files)#[1,100]

batchtools::addExperiments(algo.designs = list(RUNNER = design), repls = 1L)
#batchtools::addExperiments(algo.designs = list(RUNNER = load_study("data/study5.csv")), repls = 1L)
BBmisc::pause()

# Submit on HPC
ids = findNotDone()
#ids = findNotSubmitted()
ids = ids[, chunk := chunk(job.id, chunk.size = 1000L)]
submitJobs(ids = ids, resources = list(walltime = 60 * 60 * 24, mem = 4000))

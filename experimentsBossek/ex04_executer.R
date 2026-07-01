library(parallelMap)
library(dplyr)

set.seed(1)

load_study = function(setup.file) {
  readr::read_delim(setup.file, delim = ",")
}

# study_file = "study_data/study04_100.csv"
# out.path = "results/approximations/emoas_100"

study_file = "parameters/ex04.csv"
out.path = "results/approximations/ex04/"
if (!dir.exists(out.path))
  dir.create(out.path, recursive = TRUE)

des = load_study(study_file)

BBmisc::catf("No. of jobs: %i", nrow(des))

jobids = as.integer(des$jobid)#[1:10]

parallelMap::parallelStartMulticore(cpus = parallel::detectCores())
parallelMap::parallelLapply(jobids, function(jobid) {
  out.path = file.path(out.path, as.character(jobid))
  dir.create(out.path)
  print(out.path)
  setup.path = file.path(out.path, "setup.csv")
  write.table(des[jobid, , drop = FALSE], file = setup.path, row.names = FALSE, sep = ",", quote = FALSE)
  args = c("-OO", "ex04_runner.py", setup.path, as.character(jobid), out.path)
  args = BBmisc::collapse(args, sep = " ")
  system2("python", args)
  return(TRUE)
})
parallelMap::parallelStop()

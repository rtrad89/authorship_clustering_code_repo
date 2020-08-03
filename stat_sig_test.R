# Libraries
library(readr)
library(scmamp)
# Control variables
my.alpha = 0.05
use_b3f = F

setwd(file.path("D:", "Projects", "Authorial_Clustering_Short_Texts_nPTM",
                "Code", "authorship_clustering_code_repo"))
if(use_b3f){
  methods_performance <- read_csv(
    "__outputs__/TESTS/methods_b3f_performance.csv")

} else {
  methods_performance <- read_csv(
    "__outputs__/TESTS/methods_ari_performance.csv")
}

# Create a proper matrix
mtx_perf = as.matrix(methods_performance[,-1])
row.names(mtx_perf) = methods_performance[, 1]$Dataset

# Start the tests
if (scmamp::friedmanAlignedRanksTest(mtx_perf)$p.value < my.alpha) {
  print("Differences in performance of compared methods was found!")
  # Run Nemenyi test
  post.hoc.test = nemenyiTest(mtx_perf, alpha = my.alpha)
  # Draw the CD plot
  plotCD(mtx_perf, alpha = my.alpha, cex = 0.75)
} else {
  print("No difference in performance in the benchmark data was found.")
}

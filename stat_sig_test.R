# Libraries
library(readr)
library(scmamp)

setwd(file.path("D:", "Projects", "Authorial_Clustering_Short_Texts_nPTM",
                "Code", "authorship_clustering_code_repo"))

posthoc_test_plot = function(results_fpath, use_minimal = T, my.alpha = 0.05)
{
  methods_performance <- read_csv(results_fpath)
  if(use_minimal)
  {
    names(methods_performance)[7] = "COP_KMeans"
    names(methods_performance)[11] = "SPKMeans"
  }
  # Create a proper matrix
  mtx_perf = as.matrix(methods_performance[,-1])
  row.names(mtx_perf) = methods_performance[, 1]$Dataset

  if(use_minimal)
  {
    colsnames = c("BL_r","BL_s","BL_SOTA_le", "SPKMeans", "COP_KMeans")
    mtx_perf = mtx_perf[, colsnames]
  }


  # Start the tests
  if (scmamp::friedmanAlignedRanksTest(mtx_perf)$p.value < my.alpha) {
    print("Differences in performance of compared methods was found!")
    # Run Nemenyi test
    post.hoc.test = nemenyiTest(mtx_perf, alpha = my.alpha)
    # Draw the CD plot
    return(plotCD(mtx_perf, alpha = my.alpha, cex = 0.6))
  }
  else
  {
    print("No difference in performance in the benchmark data was found.")
  }
}

# Combine plots on one row
par(mfrow = c(1,2))
#B3F
posthoc_test_plot("__outputs__/TESTS/methods_b3f_performance.csv")
#ARI
posthoc_test_plot("__outputs__/TESTS/methods_ari_performance.csv")


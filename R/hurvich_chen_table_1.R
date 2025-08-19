#!/usr/bin/env Rscript
#
# Monte Carlo comparison of original and fixed HC implementations
#
# This script applies both the broken LongMemoryTS HC estimator and our fixed
# version to identical ARFIMA datasets, comparing results to original HC (2000)
# paper.
#
# Before running this script, you must run the Python version of the Monte
# Carlo to simulate and save the data:
#
# python examples/hurvich_chen_table_1.py --save-data
#
# This will simulate and save a large number of simulated ARFIMA(1,d,0)
# datasets to the data/hc directory, which this script will load and
# use to estimate the memory parameter.

# Load implementations
# First load the broken version and rename it
source("R/local_W_bad.R")
local.W.broken <- local.W  # Save the broken version with a new name

# Now load the fixed version, which overwrites local.W
source("R/local_W.R")
local.W.fixed <- local.W  # Optional: also save fixed version explicitly

# Parameters
N_REP <- 500
N_OBS <- 500
M <- 36

# Test cases from HC (2000) Table I
test_cases <- list(
  c(0.0, 0.0), c(0.0, 0.5), c(0.0, 0.8),
  c(0.2, 0.0), c(0.2, 0.5), c(0.2, 0.8),
  c(0.4, 0.0), c(0.4, 0.5), c(0.4, 0.8),
  c(0.6, 0.0), c(0.6, 0.5), c(0.6, 0.8),
  c(0.8, 0.0), c(0.8, 0.5), c(0.8, 0.8),
  c(1.0, 0.0), c(1.0, 0.5), c(1.0, 0.8),
  c(1.2, 0.0), c(1.2, 0.5), c(1.2, 0.8)
)

# Original HC (2000) results
paper_results <- list(
  "0.0_0.0" = list(gset_mean = -0.0013, gset_var = 0.0186),
  "0.0_0.5" = list(gset_mean =  0.0574, gset_var = 0.0188),
  "0.0_0.8" = list(gset_mean =  0.3116, gset_var = 0.0198),
  "0.2_0.0" = list(gset_mean =  0.1994, gset_var = 0.0173),
  "0.2_0.5" = list(gset_mean =  0.2580, gset_var = 0.0175),
  "0.2_0.8" = list(gset_mean =  0.5112, gset_var = 0.0190),
  "0.4_0.0" = list(gset_mean =  0.3964, gset_var = 0.0174),
  "0.4_0.5" = list(gset_mean =  0.4551, gset_var = 0.0176),
  "0.4_0.8" = list(gset_mean =  0.7091, gset_var = 0.0190),
  "0.6_0.0" = list(gset_mean =  0.5949, gset_var = 0.0173),
  "0.6_0.5" = list(gset_mean =  0.6533, gset_var = 0.0176),
  "0.6_0.8" = list(gset_mean =  0.9079, gset_var = 0.0191),
  "0.8_0.0" = list(gset_mean =  0.7945, gset_var = 0.0173),
  "0.8_0.5" = list(gset_mean =  0.8535, gset_var = 0.0174),
  "0.8_0.8" = list(gset_mean =  1.1079, gset_var = 0.0190),
  "1.0_0.0" = list(gset_mean =  0.9895, gset_var = 0.0187),
  "1.0_0.5" = list(gset_mean =  1.0488, gset_var = 0.0187),
  "1.0_0.8" = list(gset_mean =  1.2999, gset_var = 0.0171),
  "1.2_0.0" = list(gset_mean =  1.1981, gset_var = 0.0169),
  "1.2_0.5" = list(gset_mean =  1.2553, gset_var = 0.0164),
  "1.2_0.8" = list(gset_mean =  1.4453, gset_var = 0.0056)
)

# Data directory
data_dir <- "data/hc"

# Check if data exists, if not instruct user to generate it
if (!dir.exists(data_dir)) {
  cat("Data directory data/hc not found. Please run:\n")
  cat("python examples/hurvich_chen_table_1.py --save-data\n")
  cat("to generate the required datasets.\n")
  quit(status = 1)
}

cat("Hurvich and Chen (2000) Monte Carlo\n")
cat("===================================\n\n")
cat(sprintf("Sample size: n=%d, Bandwidth: m=%d, Replications: %d\n", N_OBS, M, N_REP))
cat("Comparing incorrect R LongMemoryTS and our fixed version.\n")
cat("\n")
cat("| d   | phi | HC (2000) | Orig. R  | Error  | Fixed R | Error  |\n")
cat("|-----|-----|-----------|----------|--------|---------|--------|\n")

for (case in test_cases) {
  d_true <- case[1]
  phi <- case[2]
  key <- sprintf("%.1f_%.1f", d_true, phi)

  estimates_broken <- c()
  estimates_fixed <- c()

  for (sim in 0:(N_REP-1)) {
    # Load the data file
    filename <- sprintf("%s/arfima_d%03.1f_phi%03.1f_sim%03d.dat",
                       data_dir, d_true, phi, sim)
    x <- scan(filename, quiet = TRUE)

    # Use the broken local.W function (renamed from local_W_bad.R)
    result_broken <- local.W.broken(x, m = M, taper = "HC", int = c(-1.49, 0.49))
    estimates_broken <- c(estimates_broken, result_broken$d)

    # Use the fixed local.W function (from local_W.R)
    result_fixed <- local.W.fixed(x, m = M, taper = "HC", int = c(-1.49, 0.49))
    estimates_fixed <- c(estimates_fixed, result_fixed$d)
  }

  # Compute statistics
  broken_mean <- mean(estimates_broken)
  fixed_mean <- mean(estimates_fixed)

  # Get paper result
  paper <- paper_results[[key]]
  paper_mean <- paper$gset_mean

  # Compute errors
  broken_error <- broken_mean - paper_mean
  fixed_error <- fixed_mean - paper_mean

  # Print results for this specification
  cat(sprintf("| %.1f | %.1f |  %8.4f | %8.4f | %6.3f | %7.4f | %6.3f |\n",
             d_true, phi, paper_mean, broken_mean, broken_error,
             fixed_mean, fixed_error))
}

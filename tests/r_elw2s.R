#!/usr/bin/env Rscript

# Generate test cases for two-step exact local Whittle (2ELW) estimator
# of Shimotsu (2010).

library(jsonlite)
library(longmemo)

# Source corrected LongMemoryTS functions directly
source("R/local_W.R")
source("R/ELW_est.R")

# Source local copies of LongMemoryTS files
source("/path/to/fdiff.R")

# Load datasets
nile_data <- read.csv("data/nile.csv")
sealevel_data <- read.csv("data/sealevel.csv")

# Extract series
nile <- nile_data$nile
sealevel <- sealevel_data$Sea

cat("Nile series length:", length(nile), "\n")
cat("Sea level series length:", length(sealevel), "\n")

# Test configurations
datasets <- list(
  "nile" = nile,
  "sealevel" = sealevel
)

# Different bandwidth choices
bandwidth_configs <- list(
  list(name = "small", formula = function(n) floor(n^0.60)),
  list(name = "medium", formula = function(n) floor(n^0.70)),
  list(name = "large", formula = function(n) floor(n^0.80))
)

# Taper configurations
taper_configs <- list(
  list(name = "cosine", taper = "Velasco", trend_order = 1),
  list(name = "hc", taper = "HC", trend_order = 1)
)

# Initialize results structure
results <- list()

cat("Running Two-Step Exact Local Whittle tests...\n")

for (dataset_name in names(datasets)) {
  series <- datasets[[dataset_name]]
  n <- length(series)

  cat(sprintf("Processing %s (n=%d)...\n", dataset_name, n))

  results[[dataset_name]] <- list()

  for (bw_config in bandwidth_configs) {
    bw_name <- bw_config$name
    m <- bw_config$formula(n)

    cat(sprintf("  Bandwidth %s (m=%d)...\n", bw_name, m))

    results[[dataset_name]][[bw_name]] <- list()

    for (taper_config in taper_configs) {
      taper_name <- taper_config$name
      taper_type <- taper_config$taper
      trend_order <- taper_config$trend_order

      cat(sprintf("    First step taper %s...\n", taper_name))

      result <- ELW2S(
        data = series,
        m = m,
        trend_order = trend_order,
        taper = taper_type
      )

      # Store results
      results[[dataset_name]][[bw_name]][[taper_name]] <- list(
        n = n,
        m = m,
        d_hat = result$d,
        se = result$s.e.,
        taper = taper_type,
        trend_order = if(is.na(trend_order)) NULL else trend_order
      )

      cat(sprintf("      d_hat=%.4f, se=%.4f\n", result$d, result$s.e.))

    }
  }
}

# Save results to JSON
output_file <- "tests/r_elw2s.json"
write_json(results, output_file, pretty = TRUE, auto_unbox = TRUE, digits = 15)

cat(sprintf("\nResults saved to %s\n", output_file))

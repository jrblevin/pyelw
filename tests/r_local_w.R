#!/usr/bin/env Rscript

# Generate test cases for local Whittle (Robinson, 1995) and tapered Local
# Whittle estimation (Velasco, 1999; Hurvich and Chen, 2000).

library(jsonlite)
library(longmemo)

# Source corrected LongMemoryTS functions directly
source("R/local_W.R")

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
  list(name = "none", taper = "none", diff_param = NA),
  list(name = "cosine", taper = "Velasco", diff_param = NA),
  list(name = "hc", taper = "HC", diff_param = 1)
)

# Initialize results structure
results <- list()
bounds <- c(-1.0,2.5)

cat("Running Local Whittle tests...\n")

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
      diff_param <- taper_config$diff_param

      cat(sprintf("    Taper %s...\n", taper_name))

      tryCatch({
        if (taper_name == "hc") {
          # HC taper with differencing
          result <- local.W(
            data = series,
            m = m,
            int = bounds,
            taper = taper_type,
            diff_param = diff_param
          )
        } else {
          # No taper or Velasco taper
          result <- local.W(
            data = series,
            m = m,
            int = bounds,
            taper = taper_type
          )
        }

        # Store results
        results[[dataset_name]][[bw_name]][[taper_name]] <- list(
          n = n,
          m = m,
          d_hat = result$d,
          se = result$s.e.,
          taper = taper_type,
          diff_param = if(is.na(diff_param)) NULL else diff_param
        )

        cat(sprintf("      d_hat=%.4f, se=%.4f\n", result$d, result$s.e.))

      }, error = function(e) {
        cat(sprintf("      ERROR: %s\n", e$message))
        results[[dataset_name]][[bw_name]][[taper_name]] <- list(
          error = e$message
        )
      })
    }
  }
}

# Save results to JSON
output_file <- "tests/r_local_w.json"
write_json(results, output_file, pretty = TRUE, auto_unbox = TRUE, digits = 15)

cat(sprintf("\nResults saved to %s\n", output_file))

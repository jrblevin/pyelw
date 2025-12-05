#!/usr/bin/env Rscript

# Generate test cases for LWLFC estimator of Hou and Perron (2014)

library(jsonlite)
library(longmemo)

# Source the LongMemoryTS LWLFC implementation
source("R/Hou_Perron.R")

# Modified Hou.Perron that also returns objective value and theta
Hou.Perron.full <- function(data, m) {
  lower <- c(-0.4999, 0)
  T <- length(data)
  peri <- per(data)[2:(m+1)]
  out <- optim(par=c(0,0), fn=J.M, peri=peri, m=m, T=T,
               method="L-BFGS-B", lower=lower, upper=c(0.99, 10000))
  d <- out$par[1]
  theta <- out$par[2]
  obj <- out$value
  ase <- 1/(2*sqrt(m))  # Asymptotic standard error
  return(list("d"=d, "theta"=theta, "obj"=obj, "ase"=ase))
}

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

# Different bandwidth choices (Hou and Perron recommend choices)
bandwidth_configs <- list(
  list(name = "small", formula = function(n) floor(n^0.60)),
  list(name = "medium", formula = function(n) floor(n^0.70)),
  list(name = "large", formula = function(n) floor(n^0.80))
)

# Initialize results structure
results <- list()

cat("Running LWLFC (Hou and Perron, 2014) tests...\n")

for (dataset_name in names(datasets)) {
  series <- datasets[[dataset_name]]
  n <- length(series)

  cat(sprintf("Processing %s (n=%d)...\n", dataset_name, n))

  results[[dataset_name]] <- list()

  for (bw_config in bandwidth_configs) {
    bw_name <- bw_config$name
    m <- bw_config$formula(n)

    cat(sprintf("  Bandwidth %s (m=%d)...\n", bw_name, m))

    tryCatch({
      result <- Hou.Perron.full(data = series, m = m)

      # Store results
      results[[dataset_name]][[bw_name]] <- list(
        n = n,
        m = m,
        d_hat = result$d,
        theta = result$theta,
        obj = result$obj,
        ase = result$ase # asymptotic s.e.
      )

      cat(sprintf("    d_hat=%.4f, theta=%.4f, obj=%.6f, ase=%.4f\n",
                  result$d, result$theta, result$obj, result$ase))

    }, error = function(e) {
      cat(sprintf("    ERROR: %s\n", e$message))
      results[[dataset_name]][[bw_name]] <- list(
        error = e$message
      )
    })
  }
}

# Save results to JSON
output_file <- "tests/r_lwlfc.json"
write_json(results, output_file, pretty = TRUE, auto_unbox = TRUE, digits = 15)

cat(sprintf("\nResults saved to %s\n", output_file))

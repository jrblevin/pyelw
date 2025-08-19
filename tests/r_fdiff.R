#!/usr/bin/env Rscript
#
# Generate high-precision test cases for fractional differencing using R's
# LongMemoryTS fdiff function, which implements the Jensen-Nielsen algorithm.
#

# Load the fdiff function from LongMemoryTS (archived)
source("/path/to/fdiff.R")

# Set high precision output
options(digits=16, scipen=999)

# Function to format a test case as Python dict
format_test_case <- function(name, input_vec, d_val, result_vec) {
  cat('        {\n')
  cat(sprintf('            "name": "%s",\n', name))
  cat('            "input": [')
  cat(paste(sprintf("%.15f", input_vec), collapse=", "))
  cat('],\n')
  cat(sprintf('            "d": %.15f,\n', d_val))
  cat('            "expected": [')
  cat(paste(sprintf("%.15f", result_vec), collapse=", "))
  cat(']\n')
  cat('        }')
}

cat("Generating fractional differencing test cases...\n")

test_cases <- list()

# 1. Constant series tests
constant_d_values <- c(-1.5, -1.0, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0)
constant_values <- c(1.0, 2.0, 3.5, -1.0, 0.5)
constant_lengths <- c(8, 12, 16)

for (c_val in constant_values) {
  for (n in constant_lengths) {
    for (d in constant_d_values) {
      x <- rep(c_val, n)
      result <- fdiff(x, d=d)

      name <- sprintf("constant_c%.1f_n%d_d%.1f", c_val, n, d)
      name <- gsub("-", "neg", name)
      name <- gsub("\\.", "p", name)

      test_cases[[length(test_cases)+1]] <- list(
        name = name,
        input = as.numeric(x),
        d = as.numeric(d),
        expected = as.numeric(result)
      )
    }
  }
}

# 2. Unit impulse tests
impulse_d_values <- c(-2.0, -1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5)
impulse_lengths <- c(8, 12, 16, 20)

for (n in impulse_lengths) {
  for (d in impulse_d_values) {
    x <- c(1.0, rep(0.0, n-1))
    result <- fdiff(x, d=d)

    name <- sprintf("impulse_n%d_d%.1f", n, d)
    name <- gsub("-", "neg", name)
    name <- gsub("\\.", "p", name)

    test_cases[[length(test_cases)+1]] <- list(
      name = name,
      input = as.numeric(x),
      d = as.numeric(d),
      expected = as.numeric(result)
    )
  }
}

# 3. Linear trend tests
trend_d_values <- c(-1.0, -0.5, 0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5)
trend_slopes <- c(0.5, 1.0, 2.0, -1.0, -0.5)
trend_intercepts <- c(0.0, 1.0, 5.0, -2.0)
trend_lengths <- c(8, 12, 16)

for (slope in trend_slopes) {
  for (intercept in trend_intercepts) {
    for (n in trend_lengths) {
      for (d in trend_d_values) {
        x <- intercept + slope * (1:n)
        result <- fdiff(x, d=d)

        name <- sprintf("trend_s%.1f_i%.1f_n%d_d%.1f", slope, intercept, n, d)
        name <- gsub("-", "neg", name)
        name <- gsub("\\.", "p", name)

        test_cases[[length(test_cases)+1]] <- list(
          name = name,
          input = as.numeric(x),
          d = as.numeric(d),
          expected = as.numeric(result)
        )
      }
    }
  }
}

# 4. Polynomial tests
poly_d_values <- c(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

# Quadratic: x^2
for (d in poly_d_values) {
  x <- (1:12)^2
  result <- fdiff(x, d=d)

  name <- sprintf("quadratic_d%.1f", d)
  name <- gsub("-", "neg", name)
  name <- gsub("\\.", "p", name)

  test_cases[[length(test_cases)+1]] <- list(
    name = name,
    input = as.numeric(x),
    d = as.numeric(d),
    expected = as.numeric(result)
  )
}

# Cubic: x^3
for (d in poly_d_values) {
  x <- (1:10)^3
  result <- fdiff(x, d=d)

  name <- sprintf("cubic_d%.1f", d)
  name <- gsub("-", "neg", name)
  name <- gsub("\\.", "p", name)

  test_cases[[length(test_cases)+1]] <- list(
    name = name,
    input = as.numeric(x),
    d = as.numeric(d),
    expected = as.numeric(result)
  )
}

# 5. Trigonometric tests
trig_d_values <- c(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5)

# Sine wave
for (d in trig_d_values) {
  x <- sin(2 * pi * (1:16) / 8)
  result <- fdiff(x, d=d)

  name <- sprintf("sine_d%.1f", d)
  name <- gsub("-", "neg", name)
  name <- gsub("\\.", "p", name)

  test_cases[[length(test_cases)+1]] <- list(
    name = name,
    input = as.numeric(x),
    d = as.numeric(d),
    expected = as.numeric(result)
  )
}

# Cosine wave
for (d in trig_d_values) {
  x <- cos(2 * pi * (1:16) / 8)
  result <- fdiff(x, d=d)

  name <- sprintf("cosine_d%.1f", d)
  name <- gsub("-", "neg", name)
  name <- gsub("\\.", "p", name)

  test_cases[[length(test_cases)+1]] <- list(
    name = name,
    input = as.numeric(x),
    d = as.numeric(d),
    expected = as.numeric(result)
  )
}

# 6. Exponential tests
exp_d_values <- c(-0.5, 0.0, 0.5, 1.0, 1.5)
exp_bases <- c(0.5, 0.8, 0.9, 1.1, 1.2, 1.5)

for (base in exp_bases) {
  for (d in exp_d_values) {
    x <- base^(1:12)
    result <- fdiff(x, d=d)

    name <- sprintf("exp_b%.1f_d%.1f", base, d)
    name <- gsub("-", "neg", name)
    name <- gsub("\\.", "p", name)

    test_cases[[length(test_cases)+1]] <- list(
      name = name,
      input = as.numeric(x),
      d = as.numeric(d),
      expected = as.numeric(result)
    )
  }
}

# 7. Mixed signal tests
mixed_d_values <- c(-0.5, -0.3, 0.0, 0.3, 0.4, 1.0)

# Trend + sine
for (d in mixed_d_values) {
  trend <- 1:16
  sine_wave <- 0.5 * sin(2 * pi * (1:16) / 8)
  x <- trend + sine_wave
  result <- fdiff(x, d=d)

  name <- sprintf("trend_sine_d%.1f", d)
  name <- gsub("-", "neg", name)
  name <- gsub("\\.", "p", name)

  test_cases[[length(test_cases)+1]] <- list(
    name = name,
    input = as.numeric(x),
    d = as.numeric(d),
    expected = as.numeric(result)
  )
}

# 8. Edge cases

# Very small values
for (d in c(-0.5, 0.0, 0.5, 1.0)) {
  x <- rep(1e-12, 10)
  result <- fdiff(x, d=d)

  name <- sprintf("tiny_values_d%.1f", d)
  name <- gsub("-", "neg", name)
  name <- gsub("\\.", "p", name)

  test_cases[[length(test_cases)+1]] <- list(
    name = name,
    input = as.numeric(x),
    d = as.numeric(d),
    expected = as.numeric(result)
  )
}

# Very large values
for (d in c(-0.5, 0.0, 0.5, 1.0)) {
  x <- rep(1e8, 10)
  result <- fdiff(x, d=d)

  name <- sprintf("large_values_d%.1f", d)
  name <- gsub("-", "neg", name)
  name <- gsub("\\.", "p", name)

  test_cases[[length(test_cases)+1]] <- list(
    name = name,
    input = as.numeric(x),
    d = as.numeric(d),
    expected = as.numeric(result)
  )
}

# Alternating series
for (d in c(-0.5, 0.0, 0.5, 1.0, 1.5)) {
  x <- rep(c(1, -1), 8)
  result <- fdiff(x, d=d)

  name <- sprintf("alternating_d%.1f", d)
  name <- gsub("-", "neg", name)
  name <- gsub("\\.", "p", name)

  test_cases[[length(test_cases)+1]] <- list(
    name = name,
    input = as.numeric(x),
    d = as.numeric(d),
    expected = as.numeric(result)
  )
}

# Zero with single spike
for (d in c(-0.5, 0.0, 0.5, 1.0)) {
  x <- rep(0, 12)
  x[7] <- 1.0  # Spike in middle
  result <- fdiff(x, d=d)

  name <- sprintf("spike_d%.1f", d)
  name <- gsub("-", "neg", name)
  name <- gsub("\\.", "p", name)

  test_cases[[length(test_cases)+1]] <- list(
    name = name,
    input = as.numeric(x),
    d = as.numeric(d),
    expected = as.numeric(result)
  )
}

cat(sprintf("Generated %d test cases\n", length(test_cases)))

# Save to JSON file
if (require(jsonlite, quietly=TRUE)) {
  write_json(test_cases, "r_fdiff.json",
             pretty=TRUE, digits=16, auto_unbox=TRUE)
  cat(sprintf("\nTest cases saved to: r_fdiff.json\n"))
} else {
  cat("\nNote: Install jsonlite package to save JSON output\n")
}

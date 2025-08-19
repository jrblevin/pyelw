#!/usr/bin/env Rscript
#
# Download Nelson-Plosser (1982) historical data as used in Shimotsu (2010)
#
# This script downloads the classic macroeconomic time series from Nelson and
# Plosser (1982), extended by Schotman and van Dijk (1991), which are used in
# the Shimotsu (2010) Two-Step ELW paper.
#
# We obtain this data from the `urca` package in R, which includes the extended
# Nelson-Plosser dataset as `npext`. This data set contains the fourteen U.S.
# economic time series used by Schotman and Dijk (1991).  All series are
# transformed by taking logarithms except for the bond yield. The sample period
# ends in 1988.

cat("Downloading Nelson-Plosser Historical Economic Data\n")
cat("===================================================\n")

# Check if required packages are installed
required_packages <- c("urca")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

if(length(missing_packages) > 0) {
    cat("Installing missing packages:", paste(missing_packages, collapse=", "), "\n")
    install.packages(missing_packages, repos="https://cloud.r-project.org/")
}

# Load required packages
library(urca)

# Load the Nelson-Plosser extended dataset
cat("Loading Nelson-Plosser extended dataset from urca package...\n")
data(npext)

# Display information about the dataset
cat("\nDataset information:\n")
cat("Dimensions:", dim(npext), "\n")
cat("Time period:", min(npext$year), "to", max(npext$year), "\n")

# Display series names
cat("\nSeries included:\n")
series_names <- colnames(npext)
for(i in 1:length(series_names)) {
    cat(sprintf("  %2d. %s\n", i, series_names[i]))
}

# Create output directory if it doesn't exist
if(!dir.exists("data")) {
    dir.create("data")
}

# Export the full dataset to CSV
output_file <- "data/nelson_plosser_ext.csv"
cat("\nExporting data to:", output_file, "\n")

# Write to CSV
write.csv(npext, output_file, row.names = FALSE)

cat("Successfully exported", nrow(npext), "observations of", ncol(npext)-1, "series\n")
cat("Data covers period:", min(npext$year), "to", max(npext$year), "\n")

# Display first few rows as preview
cat("\nPreview of data:\n")
print(head(npext, 5))

cat("\nData export complete!\n")

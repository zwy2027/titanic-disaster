# Install the R packages needed by main.R.
# The rocker/r-ver:4.6.1 image points CRAN at a dated snapshot (2026-10-28),
# so the same package versions are installed every time the image is built.
 
packages <- c("dplyr")
 
install.packages(packages)
 
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste("Failed to install package:", pkg))
  }
  cat(sprintf("Installed %s version %s\n", pkg, as.character(packageVersion(pkg))))
}
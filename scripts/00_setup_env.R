# ==========================================================
# Script: 00_setup_env.R
# Purpose:
#   Initialize a reproducible environment for the large-scale
#   sentiment analysis project:
#   "Why Nigerians Leave Nigeria — 2018–2025 Twitter Corpus".
#   This script:
#     - installs required packages
#     - activates renv
#     - sets up folder structure
#     - logs session info
# ==========================================================

# ---- 1. Define required packages ----
required_pkgs <- c(
  # Core data handling
  "tidyverse", "dplyr", "readr", "stringr", "lubridate",

  # NLP packages
  "tidytext", "quanteda", "quanteda.textstats", "quanteda.textmodels",

  # Visualization
  "ggplot2", "scales",

  # Project structure helpers
  "here", "fs",

  # Reproducibility
  "renv"
)

# ---- 2. Install missing packages ----
installed <- installed.packages()[, "Package"]
to_install <- setdiff(required_pkgs, installed)

if (length(to_install) > 0) {
  message("Installing missing packages: ", paste(to_install, collapse = ", "))
  install.packages(to_install, dependencies = TRUE)
}

# ---- 3. Load required packages ----
lapply(required_pkgs, library, character.only = TRUE)

# ---- 4. Initialize renv environment ----
if (!file.exists("renv.lock")) {
  message("Initializing renv for reproducibility...")
  renv::init(bare = TRUE)
} else {
  message("renv.lock found — restoring environment...")
  renv::restore(prompt = FALSE)
}

# ---- 5. Define project directories ----
paths <- list(
  raw_data       = here::here("data", "raw"),
  processed_data = here::here("data", "processed"),
  reports        = here::here("reports"),
  figures        = here::here("reports", "figures"),
  scripts        = here::here("scripts"),
  models         = here::here("models"),
  quarto         = here::here("quarto")
)

# ---- 6. Create directories if missing ----
for (p in paths) {
  if (!dir.exists(p)) {
    dir.create(p, recursive = TRUE, showWarnings = FALSE)
    message("Created folder: ", p)
  }
}

# ---- 7. Print environment summary ----
message("\nEnvironment setup complete.")
message("Project root: ", here::here())
message("R version: ", R.version.string)
message("renv activated at: ", renv::project())

# ---- 8. Save session info ----
sink("reports/session_info.txt")
sessionInfo()
sink()

message("Session info saved to reports/session_info.txt")

# ==========================================================
# Script: 01_import_schema_check.R
# Purpose: Import and validate the raw 2018–2025 Twitter corpus.
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(readr)
library(dplyr)
library(janitor)
library(lubridate)
library(here)
library(fs)

# ---- Paths ----
raw_path <- here("data", "raw", "why_nigerians_leave_2018_2025.csv")
processed_dir <- here("data", "processed")

dict_path     <- here("reports", "data_dictionary.csv")   # CHANGED from .xlsx
summary_path  <- here("reports", "import_summary.csv")
preview_path  <- here("reports", "preview_100rows.csv")
log_path      <- here("reports", "import_log.txt")

dir_create(processed_dir)

# ---- Start logging ----
cat("===== Import Log =====\n", file = log_path)
cat("Started: ", as.character(Sys.time()), "\n\n", file = log_path, append = TRUE)

# ---- Check file ----
if (!file.exists(raw_path)) {
  stop("❌ Dataset not found at: ", raw_path)
} else {
  message("📂 Found dataset: ", raw_path)
}

# ---- Import CSV ----
message("Reading dataset...")
tweets_raw <- read_csv(raw_path, show_col_types = FALSE)
tweets <- tweets_raw %>% clean_names()

message("Imported: ", nrow(tweets), " rows, ", ncol(tweets), " columns")

# ---- Schema validation ----
expected_cols <- c(
  "status_id", "created_at", "user_id", "screen_name",
  "text", "lang", "source", "reply_count", "retweet_count",
  "favorite_count", "user_followers_count", "is_retweet", "is_quote"
)

missing_cols <- setdiff(expected_cols, names(tweets))
extra_cols   <- setdiff(names(tweets), expected_cols)

cat("Missing columns: ", paste(missing_cols, collapse = ", "), "\n",
    file = log_path, append = TRUE)
cat("Unexpected columns: ", paste(extra_cols, collapse = ", "), "\n",
    file = log_path, append = TRUE)

# ---- Data dictionary (CSV only, NO writexl) ----
message("Generating data dictionary...")

data_dict <- tibble(
  column_name = names(tweets),
  data_type   = sapply(tweets, function(x) paste(class(x), collapse = ",")),
  missing_pct = sapply(tweets, function(x) {
    if (is.character(x)) {
      mean(is.na(x) | trimws(x) == "", na.rm = FALSE) * 100
    } else {
      mean(is.na(x)) * 100
    }
  }) |> round(2),
  n_unique    = sapply(tweets, n_distinct)
)


readr::write_csv(data_dict, dict_path)
cat("Data dictionary saved to: ", dict_path, "\n", file = log_path, append = TRUE)

# ---- Import summary using existing datetime ----
message("Summarizing timestamps...")

import_summary <- tibble(
  timestamp          = as.character(Sys.time()),
  total_rows         = nrow(tweets),
  unique_status_id   = n_distinct(tweets$status_id),
  duplicate_status_id = sum(duplicated(tweets$status_id)),
  missing_text       = sum(is.na(tweets$text) | tweets$text == ""),
  missing_text_pct   = round(mean(is.na(tweets$text) | tweets$text == "") * 100, 2),
  earliest_date      = min(tweets$created_at, na.rm = TRUE),   # POSIXct already valid
  latest_date        = max(tweets$created_at, na.rm = TRUE)
)

write.csv(import_summary, summary_path, row.names = FALSE)

# ---- Preview sample ----
preview <- tweets %>% slice_head(n = 100)
write.csv(preview, preview_path, row.names = FALSE)

# ---- Save imported dataset ----
saveRDS(tweets, here("data", "processed", "01_raw_imported.rds"))

# ---- Console summary ----
message("\n------ Import Summary ------")
print(import_summary)
message("----------------------------")
message("Dictionary: ", dict_path)
message("Preview: ", preview_path)
message("Log: ", log_path)
message("✅ Schema check completed successfully.")

# ---- Log completion ----
cat("Completed: ", as.character(Sys.time()), "\n",
    file = log_path, append = TRUE)

# ==========================================================
# End of script
# ==========================================================

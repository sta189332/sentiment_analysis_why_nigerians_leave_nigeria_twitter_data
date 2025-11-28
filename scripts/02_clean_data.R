# ==========================================================
# Script: 02_clean_data.R
# Purpose:
#   Clean and filter the 2018–2025 Twitter corpus for:
#   "Why Nigerians Leave Nigeria — Multi-Year Sentiment Analysis".
#
#   Key steps:
#     - load imported RDS
#     - enforce expected schema
#     - filter language
#     - remove retweets (keep original tweets)
#     - basic text normalisation
#     - add date, year, month, week variables
#     - save cleaned RDS and a summary CSV
# ==========================================================

# ---- 1. Ensure environment is initialised ----
if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(stringr)
library(lubridate)
library(here)
library(readr)

# ---- 2. Load imported data ----
raw_rds_path <- here("data", "processed", "01_raw_imported.rds")

if (!file.exists(raw_rds_path)) {
  stop("01_raw_imported.rds not found at: ", raw_rds_path,
       ". Run 01_import_schema_check.R first.")
}

tweets_raw <- readRDS(raw_rds_path)

n_raw <- nrow(tweets_raw)
message("Loaded ", n_raw, " rows from 01_raw_imported.rds")

# ---- 3. Basic schema check ----
expected_cols <- c(
  "status_id", "created_at", "user_id", "screen_name",
  "text", "lang", "source", "reply_count", "retweet_count",
  "favorite_count", "user_followers_count", "is_retweet", "is_quote"
)

if (!all(expected_cols %in% names(tweets_raw))) {
  missing <- setdiff(expected_cols, names(tweets_raw))
  warning("Missing expected columns in tweets_raw: ", paste(missing, collapse = ", "))
}

# ---- 4. Language and structural filtering ----
# For now, keep English, pidgin (pcm) and undetermined (und).
# If you later want Yoruba, Hausa, etc., adjust this vector.
keep_lang <- c("en", "pcm", "und")

tweets_struct <- tweets_raw %>%
  # language filter
  filter(lang %in% keep_lang) %>%
  # remove rows with missing or blank text
  filter(!is.na(text)) %>%
  mutate(text = str_squish(text)) %>%
  filter(text != "")

n_after_lang <- nrow(tweets_struct)
message("After language + non-empty text filter: ", n_after_lang, " rows")

# ---- 5. Remove retweets, keep only original tweets ----
tweets_orig <- tweets_struct %>%
  filter(!is_retweet) %>%
  # optional: if you want to drop quotes as well, uncomment:
  # filter(!is_quote) %>%
  distinct(status_id, .keep_all = TRUE)

n_after_orig <- nrow(tweets_orig)
message("After removing retweets and duplicates: ", n_after_orig, " rows")

# ---- 6. Text normalisation ----
# Remove URLs, @mentions, hash symbols, line breaks.
tweets_clean <- tweets_orig %>%
  mutate(
    text = str_replace_all(text, "[\r\n]", " "),
    text = str_replace_all(text, "https?://\\S+", " "),   # URLs
    text = str_replace_all(text, "@\\w+", " "),           # mentions
    text = str_replace_all(text, "#", " "),               # hash symbol
    text = str_squish(text),
    text_lower = str_to_lower(text)
  )

# ---- 7. Temporal variables ----
tweets_clean <- tweets_clean %>%
  mutate(
    date  = as_date(created_at),
    year  = year(created_at),
    month = floor_date(created_at, unit = "month"),
    week  = floor_date(created_at, unit = "week", week_start = 1)
  )

# ---- 8. Basic cleaning summary ----
clean_summary <- tibble(
  timestamp_run          = as.character(Sys.time()),
  n_raw                  = n_raw,
  n_after_lang_text      = n_after_lang,
  n_after_orig           = n_after_orig,
  n_final                = nrow(tweets_clean),
  n_lang_en              = sum(tweets_clean$lang == "en", na.rm = TRUE),
  n_lang_pcm             = sum(tweets_clean$lang == "pcm", na.rm = TRUE),
  n_lang_und             = sum(tweets_clean$lang == "und", na.rm = TRUE),
  min_date               = min(tweets_clean$created_at, na.rm = TRUE),
  max_date               = max(tweets_clean$created_at, na.rm = TRUE)
)

summary_path <- here("reports", "cleaning_summary.csv")
write_csv(clean_summary, summary_path)

# ---- 9. Save cleaned data ----
clean_rds_path <- here("data", "processed", "02_tweets_clean.rds")
saveRDS(tweets_clean, clean_rds_path)

# optional small CSV preview for manual inspection
preview_clean <- tweets_clean %>%
  select(status_id, created_at, lang, text) %>%
  slice_head(n = 100)

write_csv(preview_clean, here("reports", "preview_clean_100rows.csv"))

# ---- 10. Console report ----
message("\n------ Cleaning Summary ------")
print(clean_summary)
message("------------------------------")
message("Cleaned data saved to: ", clean_rds_path)
message("Cleaning summary saved to: ", summary_path)
message("Preview saved to: reports/preview_clean_100rows.csv")
message("✅ 02_clean_data.R completed successfully.")

# ==========================================================
# End of script
# ==========================================================

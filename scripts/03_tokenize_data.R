# ==========================================================
# Script: 03_tokenize_data.R
# Purpose:
#   Tokenise cleaned tweets into individual words, remove stopwords,
#   normalise tokens, and generate a tidy token dataset suitable
#   for sentiment analysis and topic modelling.
#
#   Input  : data/processed/02_tweets_clean.rds
#   Output : data/processed/03_tokens.rds
#            reports/token_summary.csv
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(stringr)
library(tidyr)
library(tidytext)
library(lubridate)
library(readr)
library(here)
library(fs)

# ---- Load cleaned data ----
clean_rds_path <- here("data", "processed", "02_tweets_clean.rds")

if (!file.exists(clean_rds_path)) {
  stop("02_tweets_clean.rds not found. Run 02_clean_data.R first.")
}

tweets <- readRDS(clean_rds_path)
n_docs <- nrow(tweets)
message("Loaded ", n_docs, " cleaned tweets.")

# ---- Nigerian-aware stopwords ----
# Standard stopwords
sw_std <- tidytext::stop_words

# Add custom Nigerian stopwords
sw_custom <- tibble(
  word = c(
    "na", "dey", "wey", "abi", "sha", "sha.", "una", "dem", "oga",
    "o", "oh", "eh", "u", "ur", "am", "go", "wan", "fit", "biko",
    "pls", "plz", "omo", "abeg", "e", "no", "yes", "lol", "lmao",
    "nigeria", "nigerian", "lagos", "abuja", "naija"
  ),
  lexicon = "custom"
)

stopwords_ng <- bind_rows(sw_std, sw_custom) %>%
  distinct(word)

# ---- Tokenisation ----
tokens <- tweets %>%
  select(status_id, text_lower, created_at, lang, year, month, week) %>%
  tidytext::unnest_tokens(
    word,
    text_lower,
    token = "words",
    to_lower = FALSE
  ) %>%
  filter(!word %in% stopwords_ng$word) %>%
  filter(str_detect(word, "^[a-zA-Z]+$")) %>%      # keep alphabetic only
  mutate(word = str_trim(word)) %>%
  filter(word != "")

n_tokens <- nrow(tokens)
message("Tokenisation completed: ", n_tokens, " tokens generated.")

# ---- Summary ----
token_summary <- tibble(
  run_timestamp       = as.character(Sys.time()),
  n_documents         = n_docs,
  n_tokens            = n_tokens,
  vocab_size          = n_distinct(tokens$word),
  avg_tokens_per_doc  = round(n_tokens / n_docs, 2),
  min_date            = min(tokens$created_at, na.rm = TRUE),
  max_date            = max(tokens$created_at, na.rm = TRUE),
  top_words           = paste(tokens %>%
                                count(word, sort = TRUE) %>%
                                slice_head(n = 20) %>%
                                pull(word),
                              collapse = ", ")
)

summary_path <- here("reports", "token_summary.csv")
write_csv(token_summary, summary_path)

# ---- Save tokens ----
tokens_rds_path <- here("data", "processed", "03_tokens.rds")
saveRDS(tokens, tokens_rds_path)

# ---- Console report ----
message("\n------ Tokenisation Summary ------")
print(token_summary)
message("----------------------------------")
message("Tokens saved to: ", tokens_rds_path)
message("Summary saved to: ", summary_path)
message("✅ 03_tokenize_data.R completed successfully.")

# ==========================================================
# End of script
# ==========================================================

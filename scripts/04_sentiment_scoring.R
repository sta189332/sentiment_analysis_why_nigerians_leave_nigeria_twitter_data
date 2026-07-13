# ==========================================================
# Script: 04_sentiment_scoring.R
# Purpose: Compute tweet-level AFINN, NRC, and VADER scores.
# ==========================================================

if (!"here" %in% loadedNamespaces()) source("scripts/00_setup_env.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tidytext)
  library(readr)
  library(here)
  library(tibble)
})

if (!requireNamespace("vader", quietly = TRUE)) {
  stop("Package 'vader' is required. Restore the project renv environment.")
}

vader_version <- as.character(utils::packageVersion("vader"))
if (vader_version != "0.2.1") {
  warning("Developed with vader 0.2.1; installed version: ", vader_version)
}

tweets_path <- here("data", "processed", "02_tweets_clean.rds")
tokens_path <- here("data", "processed", "03_tokens.rds")

if (!file.exists(tweets_path)) stop("Run 02_clean_data.R first.")
if (!file.exists(tokens_path)) stop("Run 03_tokenize_data.R first.")

tweets <- readRDS(tweets_path)
tokens <- readRDS(tokens_path)

required_tweets <- c(
  "status_id", "created_at", "lang", "year", "month", "week", "text",
  "reply_count", "retweet_count", "favorite_count", "user_followers_count"
)
required_tokens <- c("status_id", "word")

missing_tweets <- setdiff(required_tweets, names(tweets))
missing_tokens <- setdiff(required_tokens, names(tokens))
if (length(missing_tweets)) stop("Missing tweet columns: ", paste(missing_tweets, collapse = ", "))
if (length(missing_tokens)) stop("Missing token columns: ", paste(missing_tokens, collapse = ", "))
if (anyDuplicated(tweets$status_id)) stop("status_id must be unique in cleaned tweets.")
if (any(is.na(tweets$text) | trimws(tweets$text) == "")) stop("Missing or blank tweet text.")

# AFINN token matches and tweet-level summaries
afinn_tokens <- tokens %>%
  inner_join(tidytext::get_sentiments("afinn"), by = "word")

afinn_by_tweet <- afinn_tokens %>%
  group_by(status_id) %>%
  summarise(
    afinn_sum = sum(value, na.rm = TRUE),
    afinn_mean = mean(value, na.rm = TRUE),
    afinn_n = n(),
    .groups = "drop"
  )

saveRDS(
  afinn_tokens,
  here("data", "processed", "04_sentiment_tokens_afinn.rds")
)

# NRC category counts
nrc_tokens <- tokens %>%
  inner_join(tidytext::get_sentiments("nrc"), by = "word")

nrc_by_tweet <- nrc_tokens %>%
  count(status_id, sentiment, name = "n") %>%
  pivot_wider(
    names_from = sentiment,
    values_from = n,
    values_fill = 0,
    names_prefix = "nrc_"
  )

saveRDS(
  nrc_tokens,
  here("data", "processed", "04_sentiment_tokens_nrc.rds")
)

# Complete-tweet VADER scores
vader_raw <- vader::vader_df(
  tweets$text,
  incl_nt = TRUE,
  neu_set = TRUE,
  rm_qm = TRUE
)

required_vader <- c("compound", "pos", "neg", "neu")
missing_vader <- setdiff(required_vader, names(vader_raw))
if (length(missing_vader)) stop("Missing VADER columns: ", paste(missing_vader, collapse = ", "))
if (nrow(vader_raw) != nrow(tweets)) stop("VADER and tweet row counts differ.")

vader_by_tweet <- tibble(
  status_id = tweets$status_id,
  compound = as.numeric(vader_raw$compound),
  pos = as.numeric(vader_raw$pos),
  neg = as.numeric(vader_raw$neg),
  neu = as.numeric(vader_raw$neu)
)

if (any(!is.finite(as.matrix(vader_by_tweet[-1])))) stop("Non-finite VADER scores returned.")

# Combined tweet-level sentiment dataset
sentiment_tweets <- tweets %>%
  select(all_of(required_tweets)) %>%
  left_join(afinn_by_tweet, by = "status_id") %>%
  left_join(nrc_by_tweet, by = "status_id") %>%
  left_join(vader_by_tweet, by = "status_id")

nrc_cols <- grep("^nrc_", names(sentiment_tweets), value = TRUE)
if (!length(nrc_cols)) stop("No NRC category columns were generated.")

sentiment_tweets <- sentiment_tweets %>%
  mutate(across(all_of(nrc_cols), ~ replace_na(.x, 0L)))

if (nrow(sentiment_tweets) != nrow(tweets)) stop("Combined row count changed.")
if (anyDuplicated(sentiment_tweets$status_id)) stop("Duplicate status IDs after joins.")

sentiment_summary <- tibble(
  run_timestamp = as.character(Sys.time()),
  n_tweets = nrow(sentiment_tweets),
  n_with_afinn = sum(!is.na(sentiment_tweets$afinn_sum)),
  n_with_nrc = sum(rowSums(select(sentiment_tweets, all_of(nrc_cols))) > 0),
  n_with_vader = sum(!is.na(sentiment_tweets$compound)),
  afinn_sum_mean = mean(sentiment_tweets$afinn_sum, na.rm = TRUE),
  afinn_sum_sd = sd(sentiment_tweets$afinn_sum, na.rm = TRUE),
  vader_compound_mean = mean(sentiment_tweets$compound, na.rm = TRUE),
  vader_compound_sd = sd(sentiment_tweets$compound, na.rm = TRUE),
  vader_pos_mean = mean(sentiment_tweets$pos, na.rm = TRUE),
  vader_neg_mean = mean(sentiment_tweets$neg, na.rm = TRUE),
  vader_neu_mean = mean(sentiment_tweets$neu, na.rm = TRUE),
  min_date = min(sentiment_tweets$created_at, na.rm = TRUE),
  max_date = max(sentiment_tweets$created_at, na.rm = TRUE)
)

saveRDS(
  sentiment_tweets,
  here("data", "processed", "04_sentiment_tweets.rds")
)

write_csv(
  sentiment_summary,
  here("reports", "sentiment_summary.csv")
)

print(sentiment_summary, width = Inf)
message("04_sentiment_scoring.R completed successfully.")

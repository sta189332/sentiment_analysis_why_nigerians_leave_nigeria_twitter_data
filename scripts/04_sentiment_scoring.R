# ==========================================================
# Script: 04_sentiment_scoring.R
# Purpose:
#   Compute sentiment at:
#     (1) token level (AFINN, NRC) using 03_tokens.rds
#     (2) tweet level (VADER) using 02_tweets_clean.rds
#   and combine into a per-tweet sentiment dataset.
#
# Inputs :
#   data/processed/02_tweets_clean.rds
#   data/processed/03_tokens.rds
#
# Outputs:
#   data/processed/04_sentiment_tokens_afinn.rds
#   data/processed/04_sentiment_tokens_nrc.rds
#   data/processed/04_sentiment_tweets.rds
#   reports/sentiment_summary.csv
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(tidyr)
library(tidytext)
library(readr)
library(here)
library(lubridate)
library(vader)

# ---- 1. Load cleaned tweets and tokens ----

tweets_path <- here("data", "processed", "02_tweets_clean.rds")
tokens_path <- here("data", "processed", "03_tokens.rds")

if (!file.exists(tweets_path)) {
  stop("02_tweets_clean.rds not found. Run 02_clean_data.R first.")
}
if (!file.exists(tokens_path)) {
  stop("03_tokens.rds not found. Run 03_tokenize_data.R first.")
}

tweets <- readRDS(tweets_path)
tokens <- readRDS(tokens_path)

message("Loaded ", nrow(tweets), " cleaned tweets.")
message("Loaded ", nrow(tokens), " tokens.")


# ==========================================================
# 2. TOKEN-BASED SENTIMENT: AFINN
# ==========================================================

message("Computing token-based sentiment with AFINN...")

afinn_lex <- tidytext::get_sentiments("afinn")  # word, value

# join tokens with afinn, aggregate to tweet level
afinn_tokens <- tokens %>%
  inner_join(afinn_lex, by = c("word" = "word"))

if (nrow(afinn_tokens) == 0) {
  warning("No tokens matched AFINN lexicon. afinn_tokens will be empty.")
}

afinn_by_tweet <- afinn_tokens %>%
  group_by(status_id) %>%
  summarise(
    afinn_sum   = sum(value, na.rm = TRUE),
    afinn_mean  = mean(value, na.rm = TRUE),
    afinn_n     = n(),
    .groups = "drop"
  )

saveRDS(afinn_tokens, here("data", "processed", "04_sentiment_tokens_afinn.rds"))


# ==========================================================
# 3. TOKEN-BASED SENTIMENT: NRC (EMOTION COUNTS)
# ==========================================================

message("Computing token-based sentiment with NRC (emotions)...")

nrc_lex <- tidytext::get_sentiments("nrc")  # word, sentiment

nrc_tokens <- tokens %>%
  inner_join(nrc_lex, by = c("word" = "word"))

if (nrow(nrc_tokens) == 0) {
  warning("No tokens matched NRC lexicon. nrc_tokens will be empty.")
}

# counts of each emotion per tweet
nrc_by_tweet <- nrc_tokens %>%
  count(status_id, sentiment, name = "n") %>%
  tidyr::pivot_wider(
    names_from  = sentiment,
    values_from = n,
    values_fill = 0,
    names_prefix = "nrc_"
  )

saveRDS(nrc_tokens, here("data", "processed", "04_sentiment_tokens_nrc.rds"))


# ==========================================================
# 4. TWEET-LEVEL SENTIMENT: VADER
# ==========================================================

message("Computing tweet-level sentiment with VADER...")

library(sentimentr)

sent <- sentimentr::sentiment_by(text_input)

vader_by_tweet <- bind_cols(
  tweets %>% select(status_id),
  tibble(
    compound = sent$ave_sentiment,
    pos      = pmax(sent$ave_sentiment, 0),
    neg      = pmax(-sent$ave_sentiment, 0),
    neu      = 1 - abs(sent$ave_sentiment)
  )
)

# ==========================================================
# 5. COMBINE ALL SENTIMENT FEATURES PER TWEET
# ==========================================================

message("Combining token-based and tweet-level sentiment features...")

sentiment_tweets <- tweets %>%
  select(status_id, created_at, lang, year, month, week, text) %>%
  left_join(afinn_by_tweet, by = "status_id") %>%
  left_join(nrc_by_tweet, by = "status_id") %>%
  left_join(vader_by_tweet, by = "status_id")

# Replace NA emotion counts by zero, keep NA for afinn scores if no match
emotion_cols <- grep("^nrc_", names(sentiment_tweets), value = TRUE)

sentiment_tweets <- sentiment_tweets %>%
  mutate(across(all_of(emotion_cols), ~ ifelse(is.na(.), 0L, .)))

# ==========================================================
# 6. SUMMARISE SENTIMENT DISTRIBUTIONS
# ==========================================================

message("Summarising sentiment distributions...")

sentiment_summary <- tibble(
  run_timestamp        = as.character(Sys.time()),
  n_tweets             = nrow(sentiment_tweets),
  n_with_afinn         = sum(!is.na(sentiment_tweets$afinn_sum)),
  n_with_nrc           = sum(rowSums(select(sentiment_tweets, all_of(emotion_cols))) > 0),
  n_with_vader         = sum(!is.na(sentiment_tweets$compound)),
  afinn_sum_mean       = mean(sentiment_tweets$afinn_sum, na.rm = TRUE),
  afinn_sum_sd         = sd(sentiment_tweets$afinn_sum, na.rm = TRUE),
  vader_compound_mean  = mean(sentiment_tweets$compound, na.rm = TRUE),
  vader_compound_sd    = sd(sentiment_tweets$compound, na.rm = TRUE),
  vader_pos_mean       = mean(sentiment_tweets$pos, na.rm = TRUE),
  vader_neg_mean       = mean(sentiment_tweets$neg, na.rm = TRUE),
  vader_neu_mean       = mean(sentiment_tweets$neu, na.rm = TRUE),
  min_date             = min(sentiment_tweets$created_at, na.rm = TRUE),
  max_date             = max(sentiment_tweets$created_at, na.rm = TRUE)
)

summary_path <- here("reports", "sentiment_summary.csv")
write_csv(sentiment_summary, summary_path)

# ==========================================================
# 7. SAVE MAIN PER-TWEET SENTIMENT OBJECT
# ==========================================================

sentiment_rds_path <- here("data", "processed", "04_sentiment_tweets.rds")
saveRDS(sentiment_tweets, sentiment_rds_path)

# ---- Console report ----
message("\n------ Sentiment Summary ------")
print(sentiment_summary)
message("--------------------------------")
message("Per-tweet sentiment saved to: ", sentiment_rds_path)
message("Summary saved to: ", summary_path)
message("✅ 04_sentiment_scoring.R completed successfully.")

# ==========================================================
# End of script
# ==========================================================

# ==========================================================
# Script: 05_analysis_plots.R
# Purpose:
#   Generate descriptive analyses and visualisations for:
#     - sentiment distributions (AFINN + sentimentr polarity)
#     - temporal trends (monthly means, 2018–2025)
#     - NRC emotion profiles
#     - top positive and negative tweets
#
# Input :
#   data/processed/04_sentiment_tweets.rds
#
# Outputs:
#   data/processed/05_top_positive_tweets.csv
#   data/processed/05_top_negative_tweets.csv
#
#   outputs/figures/sentiment_distribution_afinn.png
#   outputs/figures/sentiment_distribution_compound.png
#   outputs/figures/sentiment_trend_monthly.png
#   outputs/figures/nrc_emotions_bar_total.png
#   outputs/figures/nrc_emotions_bar_rate.png
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(readr)
library(here)
library(fs)
library(stringr)

# ---- 1. Paths and loading data ----

sentiment_path <- here("data", "processed", "04_sentiment_tweets.rds")

if (!file.exists(sentiment_path)) {
  stop("04_sentiment_tweets.rds not found. Run 04_sentiment_scoring.R first.")
}

sentiment <- readRDS(sentiment_path)
n_tweets  <- nrow(sentiment)

message("Loaded sentiment dataset with ", n_tweets, " tweets.")

# Ensure date and month variables exist
if (!"date" %in% names(sentiment)) {
  sentiment <- sentiment %>%
    mutate(date = as_date(created_at))
}
if (!"month" %in% names(sentiment)) {
  sentiment <- sentiment %>%
    mutate(month = floor_date(created_at, unit = "month"))
}

# Create figures directory
fig_dir <- here("outputs", "figures")
dir_create(fig_dir)


# ==========================================================
# 2. SENTIMENT DISTRIBUTIONS
# ==========================================================

# Check required columns
if (!"afinn_sum" %in% names(sentiment)) {
  warning("Column 'afinn_sum' not found. AFINN distribution plot will be skipped.")
}
if (!"compound" %in% names(sentiment)) {
  warning("Column 'compound' not found. compound distribution plot will be skipped.")
}

# ---- 2.1 AFINN total polarity distribution ----
if ("afinn_sum" %in% names(sentiment)) {

  p_afinn <- sentiment %>%
    filter(!is.na(afinn_sum)) %>%
    ggplot(aes(x = afinn_sum)) +
    geom_histogram(binwidth = 1, alpha = 0.7) +
    geom_vline(aes(xintercept = mean(afinn_sum, na.rm = TRUE)),
               linetype = "dashed") +
    labs(
      title = "Distribution of AFINN total sentiment scores per tweet",
      x = "AFINN total sentiment score (sum of token scores)",
      y = "Tweet count"
    )

  ggsave(
    filename = file.path(fig_dir, "sentiment_distribution_afinn.png"),
    plot = p_afinn,
    width = 8,
    height = 5,
    dpi = 300
  )
}

# ---- 2.2 sentimentr compound distribution ----
if ("compound" %in% names(sentiment)) {

  p_compound <- sentiment %>%
    filter(!is.na(compound)) %>%
    ggplot(aes(x = compound)) +
    geom_histogram(binwidth = 0.1, alpha = 0.7) +
    geom_vline(aes(xintercept = mean(compound, na.rm = TRUE)),
               linetype = "dashed") +
    labs(
      title = "Distribution of sentimentr polarity (compound) per tweet",
      x = "Average sentiment (sentimentr ave_sentiment)",
      y = "Tweet count"
    )

  ggsave(
    filename = file.path(fig_dir, "sentiment_distribution_compound.png"),
    plot = p_compound,
    width = 8,
    height = 5,
    dpi = 300
  )
}


# ==========================================================
# 3. TIME-SERIES TRENDS (MONTHLY)
# ==========================================================

message("Computing monthly sentiment trends...")

monthly_trends <- sentiment %>%
  filter(!is.na(month)) %>%
  group_by(month) %>%
  summarise(
    n_tweets       = n(),
    mean_afinn     = if ("afinn_sum" %in% names(cur_data_all()))
      mean(afinn_sum, na.rm = TRUE) else NA_real_,
    mean_compound  = if ("compound" %in% names(cur_data_all()))
      mean(compound, na.rm = TRUE) else NA_real_,
    .groups = "drop"
  )

p_trend <- monthly_trends %>%
  pivot_longer(
    cols = c(mean_afinn, mean_compound),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = recode(
      metric,
      mean_afinn    = "AFINN total polarity (mean)",
      mean_compound = "sentimentr polarity (mean)"
    )
  ) %>%
  ggplot(aes(x = month, y = value, colour = metric)) +
  geom_line() +
  geom_smooth(se = FALSE, method = "loess", span = 0.3) +
  labs(
    title = "Monthly sentiment trends, 2018–2025",
    x = "Month",
    y = "Mean sentiment score",
    colour = "Metric"
  )

ggsave(
  filename = file.path(fig_dir, "sentiment_trend_monthly.png"),
  plot = p_trend,
  width = 9,
  height = 5,
  dpi = 300
)


# ==========================================================
# 4. NRC EMOTION VISUALISATIONS
# ==========================================================

message("Summarising NRC emotions...")

emotion_cols <- grep("^nrc_", names(sentiment), value = TRUE)

if (length(emotion_cols) == 0) {
  warning("No NRC emotion columns found (nrc_*). Skipping emotion plots.")
} else {

  # Total counts per emotion
  nrc_totals <- sentiment %>%
    select(all_of(emotion_cols)) %>%
    summarise(across(everything(), sum, na.rm = TRUE)) %>%
    pivot_longer(everything(), names_to = "emotion", values_to = "count") %>%
    mutate(
      emotion = str_remove(emotion, "^nrc_"),
      emotion = factor(emotion, levels = emotion[order(-count)])
    )

  p_nrc_total <- nrc_totals %>%
    ggplot(aes(x = emotion, y = count)) +
    geom_col() +
    labs(
      title = "Total NRC emotion token counts across all tweets",
      x = "Emotion",
      y = "Total count"
    )

  ggsave(
    filename = file.path(fig_dir, "nrc_emotions_bar_total.png"),
    plot = p_nrc_total,
    width = 8,
    height = 5,
    dpi = 300
  )

  # Emotion rates per tweet (mean count per tweet)
  nrc_rates <- sentiment %>%
    select(all_of(emotion_cols)) %>%
    summarise(across(everything(), ~ mean(., na.rm = TRUE))) %>%
    pivot_longer(everything(), names_to = "emotion", values_to = "mean_per_tweet") %>%
    mutate(
      emotion = str_remove(emotion, "^nrc_"),
      emotion = factor(emotion, levels = emotion[order(-mean_per_tweet)])
    )

  p_nrc_rate <- nrc_rates %>%
    ggplot(aes(x = emotion, y = mean_per_tweet)) +
    geom_col() +
    labs(
      title = "Average NRC emotion tokens per tweet",
      x = "Emotion",
      y = "Mean tokens per tweet"
    )

  ggsave(
    filename = file.path(fig_dir, "nrc_emotions_bar_rate.png"),
    plot = p_nrc_rate,
    width = 8,
    height = 5,
    dpi = 300
  )
}


# ==========================================================
# 5. TOP POSITIVE AND NEGATIVE TWEETS
# ==========================================================

message("Extracting top positive and negative tweets...")

# Use sentimentr polarity (compound) if available, otherwise AFINN
if ("compound" %in% names(sentiment)) {
  score_col <- "compound"
} else if ("afinn_sum" %in% names(sentiment)) {
  score_col <- "afinn_sum"
} else {
  stop("No compound or afinn_sum column found for ranking tweets.")
}

sentiment_ranked <- sentiment %>%
  filter(!is.na(.data[[score_col]])) %>%
  select(status_id, created_at, lang, text, !!score_col) %>%
  rename(sentiment_score = !!score_col)

top_positive <- sentiment_ranked %>%
  arrange(desc(sentiment_score)) %>%
  slice_head(n = 100)

top_negative <- sentiment_ranked %>%
  arrange(sentiment_score) %>%
  slice_head(n = 100)

# Save as CSV for inspection and potential quotation in the paper
pos_path <- here("data", "processed", "05_top_positive_tweets.csv")
neg_path <- here("data", "processed", "05_top_negative_tweets.csv")

write_csv(top_positive, pos_path)
write_csv(top_negative, neg_path)


# ==========================================================
# 6. SUMMARY TO CONSOLE
# ==========================================================

message("\n------ 05_analysis_plots Summary ------")
message("Tweets analysed: ", n_tweets)
if ("afinn_sum" %in% names(sentiment)) {
  message("AFINN mean: ", round(mean(sentiment$afinn_sum, na.rm = TRUE), 3))
}
if ("compound" %in% names(sentiment)) {
  message("sentimentr compound mean: ", round(mean(sentiment$compound, na.rm = TRUE), 3))
}
if (length(emotion_cols) > 0) {
  message("NRC emotions available: ", paste(str_remove(emotion_cols, '^nrc_'), collapse = ", "))
}
message("Top positive tweets saved to: ", pos_path)
message("Top negative tweets saved to: ", neg_path)
message("Figures written to: ", fig_dir)
message("✅ 05_analysis_plots.R completed successfully.")
# ==========================================================
# End of script
# ==========================================================

# ==========================================================
# Script: 09B_sentiment_engagement.R
# Purpose:
#   Analyse relationship between sentiment and engagement metrics:
#     - retweet_count
#     - favorite_count
#     - reply_count
#     - user_followers_count
#
#   Outputs descriptive summaries, correlations, regressions,
#   and visualisations.
#
# Input:
#   data/processed/04_sentiment_tweets.rds
#
# Outputs:
#   reports/sentiment_engagement_summary.csv
#   reports/sentiment_engagement_correlations.csv
#   reports/viral_negative_tweets_top200.csv
#   outputs/figures/engagement_vs_compound.png
#   outputs/figures/engagement_vs_afinn.png
#   outputs/figures/engagement_hexbin_compound.png
#   outputs/figures/engagement_hexbin_afinn.png
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(ggplot2)
library(readr)
library(here)
library(tidyr)
library(scales)

# ---- 1. Load sentiment dataset ----

sent_path <- here("data", "processed", "04_sentiment_tweets.rds")
if (!file.exists(sent_path)) {
  stop("04_sentiment_tweets.rds not found. Run 04_sentiment_scoring.R first.")
}

sent <- readRDS(sent_path)

required_cols <- c(
  "compound", "afinn_sum",
  "retweet_count", "favorite_count", "reply_count",
  "user_followers_count", "status_id", "text"
)

if (!all(required_cols %in% names(sent))) {
  stop("Some required engagement/sentiment columns are missing.")
}

message("Loaded sentiment dataset with ", nrow(sent), " tweets.")

# ---- 2. Basic engagement statistics ----

sentiment_engage_summary <- sent %>%
  summarise(
    n_tweets = n(),
    mean_retweets = mean(retweet_count, na.rm = TRUE),
    median_retweets = median(retweet_count, na.rm = TRUE),
    mean_favorites = mean(favorite_count, na.rm = TRUE),
    median_favorites = median(favorite_count, na.rm = TRUE),
    mean_replies = mean(reply_count, na.rm = TRUE),
    median_replies = median(reply_count, na.rm = TRUE),
    mean_followers = mean(user_followers_count, na.rm = TRUE),
    median_followers = median(user_followers_count, na.rm = TRUE)
  )

summary_path <- here("reports", "sentiment_engagement_summary.csv")
write_csv(sentiment_engage_summary, summary_path)
message("Saved engagement summary to: ", summary_path)

# ---- 3. Correlation analysis ----

corr_df <- sent %>%
  select(
    compound, afinn_sum,
    retweet_count, favorite_count, reply_count, user_followers_count
  )

corr_matrix <- cor(corr_df, use = "pairwise.complete.obs", method = "spearman")

corr_path <- here("reports", "sentiment_engagement_correlations.csv")
write_csv(as.data.frame(corr_matrix), corr_path)
message("Saved correlation matrix to: ", corr_path)

message("\nCorrelation matrix (Spearman):")
print(round(corr_matrix, 3))

# ---- 4. Visualisations ----

fig_dir <- here("outputs", "figures")
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

# 4.1 sentimentr compound vs engagement
p_compound_engage <- ggplot(sent, aes(x = compound, y = retweet_count)) +
  geom_point(alpha = 0.05) +
  scale_y_log10(labels = comma) +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(
    title = "Retweets vs Compound Sentiment (log-scale)",
    x = "Compound sentiment",
    y = "Retweet count (log scale)"
  )

compound_fig <- file.path(fig_dir, "engagement_vs_compound.png")
ggsave(compound_fig, p_compound_engage, width = 8, height = 5, dpi = 300)

# 4.2 afinn vs engagement
p_afinn_engage <- ggplot(sent, aes(x = afinn_sum, y = retweet_count)) +
  geom_point(alpha = 0.05) +
  scale_y_log10(labels = comma) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(
    title = "Retweets vs AFINN Sentiment (log-scale)",
    x = "AFINN sentiment",
    y = "Retweet count (log scale)"
  )

afinn_fig <- file.path(fig_dir, "engagement_vs_afinn.png")
ggsave(afinn_fig, p_afinn_engage, width = 8, height = 5, dpi = 300)

# 4.3 Hexbin plots (better for large data)

p_hex_compound <- ggplot(sent, aes(x = compound, y = retweet_count)) +
  geom_hex(bins = 40) +
  scale_fill_viridis_c(option = "plasma") +
  scale_y_log10(labels = comma) +
  labs(
    title = "Retweets vs Compound Sentiment (Hexbin)",
    x = "Compound sentiment",
    y = "Retweet count (log scale)"
  )

hex_compound_fig <- file.path(fig_dir, "engagement_hexbin_compound.png")
ggsave(hex_compound_fig, p_hex_compound, width = 8, height = 5, dpi = 300)

p_hex_afinn <- ggplot(sent, aes(x = afinn_sum, y = retweet_count)) +
  geom_hex(bins = 40) +
  scale_fill_viridis_c(option = "magma") +
  scale_y_log10(labels = comma) +
  labs(
    title = "Retweets vs AFINN Sentiment (Hexbin)",
    x = "AFINN sentiment",
    y = "Retweet count (log scale)"
  )

hex_afinn_fig <- file.path(fig_dir, "engagement_hexbin_afinn.png")
ggsave(hex_afinn_fig, p_hex_afinn, width = 8, height = 5, dpi = 300)

# ---- 5. Identify viral negative tweets ----

viral_negative <- sent %>%
  filter(afinn_sum < 0 | compound < 0) %>%
  arrange(desc(retweet_count)) %>%
  slice_head(n = 200) %>%
  select(status_id, text, retweet_count, favorite_count, compound, afinn_sum)

viral_path <- here("data", "processed", "viral_negative_tweets_top200.csv")
write_csv(viral_negative, viral_path)
message("Saved viral negative tweets to: ", viral_path)

# ---- 6. Console summary ----

message("\n------ 09B_sentiment_engagement Summary ------")
print(sentiment_engage_summary)

message("\nTop correlations (Spearman):")
print(round(corr_matrix, 3))

message("Figures saved to: ", fig_dir)
message("Data saved to: ", viral_path)
message("Reports saved to: ", summary_path, " and ", corr_path)
message("✅ 09B_sentiment_engagement.R completed successfully.")
# ==========================================================
# End of script
# ==========================================================

# ==========================================================
# Script: 09A_sentiment_by_language.R
# Purpose:
#   Analyse sentiment patterns by tweet language
#   (e.g. en vs pcm vs other).
#
# Input:
#   data/processed/04_sentiment_tweets.rds
#
# Outputs:
#   reports/sentiment_by_language_summary.csv
#   reports/nrc_by_language_summary.csv
#   outputs/figures/lang_compound_boxplot.png
#   outputs/figures/lang_afinn_boxplot.png
#   outputs/figures/lang_nrc_emotions_bar.png
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(tidyr)
library(ggplot2)
library(here)
library(readr)
library(stringr)
library(forcats)

# ---- 1. Load sentiment dataset ----

sent_path <- here("data", "processed", "04_sentiment_tweets.rds")
if (!file.exists(sent_path)) {
  stop("04_sentiment_tweets.rds not found. Run 04_sentiment_scoring.R first.")
}

sent <- readRDS(sent_path)

if (!all(c("lang", "afinn_sum", "compound") %in% names(sent))) {
  stop("Expected columns 'lang', 'afinn_sum', 'compound' are missing in sentiment object.")
}

message("Loaded sentiment dataset with ", nrow(sent), " tweets.")

# ---- 2. Inspect and group languages ----

lang_counts <- sent %>%
  count(lang, sort = TRUE)

message("Language distribution:")
print(lang_counts)

# Keep top 3 languages explicitly; others go to "other"
top_n_langs <- 3L

top_langs <- lang_counts %>%
  slice_head(n = top_n_langs) %>%
  pull(lang)

sent <- sent %>%
  mutate(
    lang_group = case_when(
      is.na(lang) ~ "other",
      lang %in% top_langs ~ lang,
      TRUE ~ "other"
    ),
    lang_group = factor(lang_group, levels = c(top_langs, "other"))
  )

# ---- 3. Sentiment summary by language ----

sentiment_by_lang <- sent %>%
  group_by(lang_group) %>%
  summarise(
    n_tweets         = n(),
    mean_afinn       = mean(afinn_sum, na.rm = TRUE),
    sd_afinn         = sd(afinn_sum, na.rm = TRUE),
    mean_compound    = mean(compound, na.rm = TRUE),
    sd_compound      = sd(compound, na.rm = TRUE),
    median_afinn     = median(afinn_sum, na.rm = TRUE),
    median_compound  = median(compound, na.rm = TRUE),
    prop_afinn_neg   = mean(afinn_sum < 0, na.rm = TRUE),
    prop_comp_neg    = mean(compound < 0, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(n_tweets))

summary_path <- here("reports", "sentiment_by_language_summary.csv")
write_csv(sentiment_by_lang, summary_path)

message("Sentiment by language summary written to: ", summary_path)

# ---- 4. NRC emotion distribution by language ----

emotion_cols <- grep("^nrc_", names(sent), value = TRUE)

if (length(emotion_cols) == 0) {
  warning("No NRC emotion columns found (no '^nrc_' variables). Skipping NRC by language summary.")
  nrc_by_lang <- tibble()
} else {

  # Convert counts to presence indicators for per tweet
  nrc_long <- sent %>%
    select(lang_group, all_of(emotion_cols)) %>%
    pivot_longer(
      cols      = all_of(emotion_cols),
      names_to  = "emotion",
      values_to = "count"
    ) %>%
    mutate(
      emotion = str_replace(emotion, "^nrc_", ""),
      present = if_else(is.na(count) | count <= 0, 0L, 1L)
    )

  # Emotion prevalence per language
  nrc_by_lang <- nrc_long %>%
    group_by(lang_group, emotion) %>%
    summarise(
      n_tweets          = n(),
      tweets_with_emotion = sum(present, na.rm = TRUE),
      prop_with_emotion = tweets_with_emotion / n_tweets,
      mean_count        = mean(count, na.rm = TRUE),
      .groups = "drop"
    )

  nrc_path <- here("reports", "nrc_by_language_summary.csv")
  write_csv(nrc_by_lang, nrc_path)
  message("NRC emotions by language summary written to: ", nrc_path)
}

# ---- 5. Visualisations ----

fig_dir <- here("outputs", "figures")
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

# 5.1 Boxplot of compound sentiment by language
p_compound <- ggplot(
  sent %>% filter(!is.na(lang_group)),
  aes(x = lang_group, y = compound)
) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_boxplot(outlier.alpha = 0.2) +
  labs(
    title = "Compound sentiment by language group",
    x = "Language group",
    y = "sentimentr compound"
  )

compound_fig <- file.path(fig_dir, "lang_compound_boxplot.png")
ggsave(compound_fig, p_compound, width = 7, height = 5, dpi = 300)
message("Saved compound-by-language boxplot to: ", compound_fig)

# 5.2 Boxplot of AFINN sentiment by language
p_afinn <- ggplot(
  sent %>% filter(!is.na(lang_group)),
  aes(x = lang_group, y = afinn_sum)
) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_boxplot(outlier.alpha = 0.2) +
  labs(
    title = "AFINN sentiment by language group",
    x = "Language group",
    y = "AFINN total score"
  )

afinn_fig <- file.path(fig_dir, "lang_afinn_boxplot.png")
ggsave(afinn_fig, p_afinn, width = 7, height = 5, dpi = 300)
message("Saved AFINN-by-language boxplot to: ", afinn_fig)

# 5.3 NRC emotions bar plot by language (if available)
if (length(emotion_cols) > 0 && nrow(nrc_by_lang) > 0) {

  p_nrc <- nrc_by_lang %>%
    ggplot(aes(x = emotion, y = prop_with_emotion, fill = lang_group)) +
    geom_col(position = "dodge") +
    labs(
      title = "NRC emotion prevalence by language group",
      x = "Emotion",
      y = "Proportion of tweets with emotion"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  nrc_fig <- file.path(fig_dir, "lang_nrc_emotions_bar.png")
  ggsave(nrc_fig, p_nrc, width = 9, height = 5, dpi = 300)
  message("Saved NRC emotion-by-language bar plot to: ", nrc_fig)
}

# ---- 6. Console summary ----

message("\n------ 09A_sentiment_by_language Summary ------")
print(sentiment_by_lang)
if (exists("nrc_by_lang") && nrow(nrc_by_lang) > 0) {
  message("\nNRC emotion prevalence (head):")
  print(head(nrc_by_lang, 20))
}
message("Figures directory: ", fig_dir)
message("✅ 09A_sentiment_by_language.R completed successfully.")
# ==========================================================
# End of script
# ==========================================================

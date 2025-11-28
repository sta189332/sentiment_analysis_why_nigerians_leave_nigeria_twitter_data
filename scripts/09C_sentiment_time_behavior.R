# ==========================================================
# Script: 09C_sentiment_time_behavior.R
# Purpose:
#   Analyse temporal patterns in sentiment:
#     - Hour of day
#     - Day of week
#   Optionally:
#     - Account age effects if account_created_at is available
#
# Input:
#   data/processed/04_sentiment_tweets.rds
#
# Outputs:
#   reports/sentiment_by_hour.csv
#   reports/sentiment_by_weekday.csv
#   reports/sentiment_by_hour_lang.csv
#   reports/sentiment_by_account_age.csv  (only if account_created_at exists)
#
#   outputs/figures/hourly_compound_line.png
#   outputs/figures/hourly_afinn_line.png
#   outputs/figures/weekday_compound_boxplot.png
#   outputs/figures/hour_lang_compound_heatmap.png
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(ggplot2)
library(lubridate)
library(here)
library(readr)
library(tidyr)
library(forcats)

# ---- 1. Load sentiment dataset ----

sent_path <- here("data", "processed", "04_sentiment_tweets.rds")
if (!file.exists(sent_path)) {
  stop("04_sentiment_tweets.rds not found. Run 04_sentiment_scoring.R first.")
}

sent <- readRDS(sent_path)
message("Loaded sentiment dataset with ", nrow(sent), " tweets.")

if (!all(c("created_at", "compound", "afinn_sum", "lang") %in% names(sent))) {
  stop("Required columns created_at, compound, afinn_sum, lang are missing.")
}

# ---- 2. Derive time-of-day and weekday variables ----

sent_time <- sent %>%
  mutate(
    hour      = lubridate::hour(created_at),
    weekday   = lubridate::wday(created_at, label = TRUE, abbr = TRUE, week_start = 1),
    # Language grouping reused from 09A
    lang_group = case_when(
      is.na(lang) ~ "other",
      lang %in% c("en", "pcm") ~ lang,
      TRUE ~ "other"
    ),
    lang_group = factor(lang_group, levels = c("en", "pcm", "other"))
  )

# ---- 3. Sentiment by hour of day ----

sent_by_hour <- sent_time %>%
  group_by(hour) %>%
  summarise(
    n_tweets       = n(),
    mean_compound  = mean(compound, na.rm = TRUE),
    sd_compound    = sd(compound, na.rm = TRUE),
    mean_afinn     = mean(afinn_sum, na.rm = TRUE),
    sd_afinn       = sd(afinn_sum, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(hour)

hour_path <- here("reports", "sentiment_by_hour.csv")
write_csv(sent_by_hour, hour_path)
message("Saved hourly sentiment summary to: ", hour_path)

# ---- 4. Sentiment by weekday ----

sent_by_weekday <- sent_time %>%
  group_by(weekday) %>%
  summarise(
    n_tweets       = n(),
    mean_compound  = mean(compound, na.rm = TRUE),
    sd_compound    = sd(compound, na.rm = TRUE),
    mean_afinn     = mean(afinn_sum, na.rm = TRUE),
    sd_afinn       = sd(afinn_sum, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(weekday)

weekday_path <- here("reports", "sentiment_by_weekday.csv")
write_csv(sent_by_weekday, weekday_path)
message("Saved weekday sentiment summary to: ", weekday_path)

# ---- 5. Sentiment by hour × language ----

sent_by_hour_lang <- sent_time %>%
  group_by(hour, lang_group) %>%
  summarise(
    n_tweets      = n(),
    mean_compound = mean(compound, na.rm = TRUE),
    mean_afinn    = mean(afinn_sum, na.rm = TRUE),
    .groups = "drop"
  )

hour_lang_path <- here("reports", "sentiment_by_hour_lang.csv")
write_csv(sent_by_hour_lang, hour_lang_path)
message("Saved sentiment by hour × language to: ", hour_lang_path)

# ---- 6. Optional: Account age, if available ----

account_age_path <- here("reports", "sentiment_by_account_age.csv")

if ("account_created_at" %in% names(sent)) {
  message("account_created_at detected. Computing account age effects.")

  sent_age <- sent %>%
    mutate(
      account_age_days = as.numeric(difftime(created_at, account_created_at, units = "days")),
      account_age_years = account_age_days / 365.25
    )

  sent_by_age <- sent_age %>%
    filter(!is.na(account_age_years), account_age_years >= 0) %>%
    mutate(
      age_band = cut(
        account_age_years,
        breaks = c(0, 1, 2, 3, 5, 10, Inf),
        labels = c("0–1y", "1–2y", "2–3y", "3–5y", "5–10y", ">10y"),
        right = FALSE
      )
    ) %>%
    group_by(age_band) %>%
    summarise(
      n_tweets      = n(),
      mean_compound = mean(compound, na.rm = TRUE),
      mean_afinn    = mean(afinn_sum, na.rm = TRUE),
      .groups = "drop"
    )

  write_csv(sent_by_age, account_age_path)
  message("Saved account age sentiment summary to: ", account_age_path)

} else {
  message("No account_created_at column found. Skipping account age analysis.")
}

# ---- 7. Figures directory ----

fig_dir <- here("outputs", "figures")
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

# ---- 7.1 Line plot: mean compound by hour ----

p_hour_compound <- ggplot(sent_by_hour, aes(x = hour, y = mean_compound)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_line(size = 1) +
  geom_point(size = 1.5) +
  scale_x_continuous(breaks = 0:23) +
  labs(
    title = "Mean compound sentiment by hour of day",
    x = "Hour of day (0–23)",
    y = "Mean compound sentiment"
  )

hour_compound_fig <- file.path(fig_dir, "hourly_compound_line.png")
ggsave(hour_compound_fig, p_hour_compound, width = 8, height = 5, dpi = 300)
message("Saved hourly compound sentiment plot to: ", hour_compound_fig)

# ---- 7.2 Line plot: mean AFINN by hour ----

p_hour_afinn <- ggplot(sent_by_hour, aes(x = hour, y = mean_afinn)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_line(size = 1, color = "red") +
  geom_point(size = 1.5, color = "red") +
  scale_x_continuous(breaks = 0:23) +
  labs(
    title = "Mean AFINN sentiment by hour of day",
    x = "Hour of day (0–23)",
    y = "Mean AFINN score"
  )

hour_afinn_fig <- file.path(fig_dir, "hourly_afinn_line.png")
ggsave(hour_afinn_fig, p_hour_afinn, width = 8, height = 5, dpi = 300)
message("Saved hourly AFINN sentiment plot to: ", hour_afinn_fig)

# ---- 7.3 Boxplot: compound sentiment by weekday ----

p_weekday_compound <- ggplot(sent_time, aes(x = weekday, y = compound)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_boxplot(outlier.alpha = 0.15) +
  labs(
    title = "Compound sentiment by day of week",
    x = "Day of week",
    y = "Compound sentiment"
  )

weekday_compound_fig <- file.path(fig_dir, "weekday_compound_boxplot.png")
ggsave(weekday_compound_fig, p_weekday_compound, width = 7, height = 5, dpi = 300)
message("Saved weekday compound sentiment boxplot to: ", weekday_compound_fig)

# ---- 7.4 Heatmap: hour × language, mean compound ----

p_hour_lang_heat <- ggplot(
  sent_by_hour_lang,
  aes(x = hour, y = lang_group, fill = mean_compound)
) +
  geom_tile() +
  scale_x_continuous(breaks = 0:23) +
  scale_fill_viridis_c(option = "plasma") +
  labs(
    title = "Mean compound sentiment by hour and language",
    x = "Hour of day",
    y = "Language group",
    fill = "Mean compound"
  )

hour_lang_fig <- file.path(fig_dir, "hour_lang_compound_heatmap.png")
ggsave(hour_lang_fig, p_hour_lang_heat, width = 9, height = 4, dpi = 300)
message("Saved hour × language compound heatmap to: ", hour_lang_fig)

# ---- 8. Console summary ----

message("\n------ 09C_sentiment_time_behavior Summary ------")
message("Hourly sentiment (head):")
print(head(sent_by_hour, 24))

message("\nWeekday sentiment (all):")
print(sent_by_weekday)

if (file.exists(account_age_path)) {
  message("\nAccount age summary path: ", account_age_path)
} else {
  message("\nAccount age summary not produced (no account_created_at).")
}

message("Figures directory: ", fig_dir)
message("✅ 09C_sentiment_time_behavior.R completed successfully.")
# ==========================================================
# End of script
# ==========================================================

# ==========================================================
# Script: 09G_sentiment_calendar.R
# Purpose: Create day-by-day sentiment calendar visualisations.
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(lubridate)
library(ggplot2)
library(here)
library(readr)
library(tidyr)
library(scales)

# ==========================================================
# 1. Load sentiment-scored tweets
# ==========================================================

sent <- readRDS(here("data/processed/04_sentiment_tweets.rds"))

message("Loaded sentiment dataset: ", nrow(sent))

# Ensure a date column exists
sent <- sent %>%
  mutate(date = as.Date(created_at))

# ==========================================================
# 2. Aggregate daily sentiment
# ==========================================================

daily_sent <- sent %>%
  group_by(date) %>%
  summarise(
    mean_compound   = mean(compound, na.rm = TRUE),
    prop_negative   = mean(compound < 0, na.rm = TRUE),
    n_tweets        = n()
  ) %>%
  ungroup()

write_csv(
  daily_sent,
  here("reports/sentiment_calendar_daily.csv")
)

message("Daily sentiment file written: ",
        here("reports/sentiment_calendar_daily.csv"))

# ==========================================================
# 3. Calendar Heatmap - Average Compound Score
# ==========================================================

p1 <- daily_sent %>%
  mutate(
    year  = year(date),
    month = month(date, label = TRUE),
    day   = day(date)
  ) %>%
  ggplot(aes(x = day, y = month, fill = mean_compound)) +
  geom_tile(color = "white", linewidth = 0.3) +
  scale_fill_gradient2(
    low = "red",
    mid = "white",
    high = "blue",
    midpoint = 0,
    name = "Avg Compound"
  ) +
  facet_wrap(~year, ncol = 1) +
  theme_minimal(base_size = 11) +
  labs(
    title = "Sentiment Calendar: Average Compound Score",
    x = "Day of Month",
    y = "Month"
  )

ggsave(
  here("outputs/figures/sentiment_calendar_mean_compound.png"),
  p1, width = 10, height = 12, dpi = 300
)

message("Saved compound calendar heatmap → ",
        here("outputs/figures/sentiment_calendar_mean_compound.png"))

# ==========================================================
# 4. Calendar Heatmap - Proportion of Negative Tweets
# ==========================================================

p2 <- daily_sent %>%
  mutate(
    year  = year(date),
    month = month(date, label = TRUE),
    day   = day(date)
  ) %>%
  ggplot(aes(x = day, y = month, fill = prop_negative)) +
  geom_tile(color = "white", linewidth = 0.3) +
  scale_fill_gradient(
    low = "white",
    high = "red",
    labels = percent_format(accuracy = 1),
    name = "Proportion Negative"
  ) +
  facet_wrap(~year, ncol = 1) +
  theme_minimal(base_size = 11) +
  labs(
    title = "Sentiment Calendar: Proportion of Negative Tweets",
    x = "Day of Month",
    y = "Month"
  )

ggsave(
  here("outputs/figures/sentiment_calendar_prop_negative.png"),
  p2, width = 10, height = 12, dpi = 300
)

message("Saved negative proportion calendar heatmap → ",
        here("outputs/figures/sentiment_calendar_prop_negative.png"))

message("✅ 09G_sentiment_calendar.R completed successfully.")

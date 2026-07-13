# ------------------------------------------------------------
# Script: 06_structural_breaks.R
# Purpose: Identify structural breaks in daily compound sentiment.
# ------------------------------------------------------------

library(dplyr)
library(ggplot2)
library(strucchange)
library(readr)
library(here)

message("=== 06: Structural Breaks Analysis ===")

# Load daily sentiment
sent <- readRDS(here("data/processed/04_sentiment_tweets.rds"))

df <- sent %>%
  mutate(date = as.Date(created_at)) %>%
  group_by(date) %>%
  summarise(mean_compound = mean(compound, na.rm = TRUE)) %>%
  ungroup()

# Breakpoints
bp <- breakpoints(mean_compound ~ 1, data = df)

bp_df <- tibble(
  break_id = seq_along(bp$breakpoints),
  index = bp$breakpoints,
  date = df$date[bp$breakpoints]
)

write_csv(bp_df, here("reports/structural_breaks_summary.csv"))

# Plot
p <- ggplot(df, aes(date, mean_compound)) +
  geom_line(color = "steelblue") +
  geom_vline(
    xintercept = bp_df$date,
    color = "red", linetype = "dashed"
  ) +
  labs(
    title = "Structural Breaks in Daily Compound Sentiment",
    x = "Date",
    y = "Mean Compound Sentiment"
  ) +
  theme_minimal()

ggsave(
  here("outputs/figures/breakpoints_compound.png"),
  p, width = 10, height = 5
)

message("Structural break analysis completed.")

# ==========================================================
# Script: 06_structural_breaks.R
# Purpose:
#   Detect structural breaks in sentiment time series
#   using breakpoints() and segmented regression.
#
# Input:
#   data/processed/04_sentiment_tweets.rds
#
# Outputs:
#   outputs/figures/breakpoints_compound.png
#   outputs/figures/breakpoints_afinn.png
#   reports/structural_breaks_summary.csv
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(strucchange)
library(ggplot2)
library(here)
library(readr)
library(lubridate)
library(tsibble)

sent_path <- here("data", "processed", "04_sentiment_tweets.rds")
sent <- readRDS(sent_path)

# ensure monthly aggregation
monthly <- sent %>%
  mutate(month = floor_date(created_at, "month")) %>%
  group_by(month) %>%
  summarise(
    mean_compound = mean(compound, na.rm = TRUE),
    mean_afinn    = mean(afinn_sum, na.rm = TRUE),
    .groups = "drop"
  )

# ==========================================================
# 1. STRUCTURAL BREAKS FOR VADER COMPOUND
# ==========================================================

bp_comp <- breakpoints(mean_compound ~ 1, data = monthly)

bp_comp_breaks <- breakdates(bp_comp)
bp_comp_breaks

# Plot
p_comp <- ggplot(monthly, aes(x = month, y = mean_compound)) +
  geom_line() +
  geom_vline(xintercept = bp_comp_breaks, color = "red", linetype = "dashed") +
  labs(
    title = "Structural Breaks in Sentiment (sentimentr compound)",
    x = "Month",
    y = "Mean compound sentiment"
  )

fig_dir <- here("outputs", "figures")
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

ggsave(
  filename = file.path(fig_dir, "breakpoints_compound.png"),
  plot = p_comp, width = 9, height = 5, dpi = 300
)

# ==========================================================
# 2. STRUCTURAL BREAKS FOR AFINN
# ==========================================================

bp_af <- breakpoints(mean_afinn ~ 1, data = monthly)
bp_af_breaks <- breakdates(bp_af)

p_af <- ggplot(monthly, aes(x = month, y = mean_afinn)) +
  geom_line() +
  geom_vline(xintercept = bp_af_breaks, color = "blue", linetype = "dashed") +
  labs(
    title = "Structural Breaks in Sentiment (AFINN total score)",
    x = "Month",
    y = "Mean AFINN sentiment"
  )

ggsave(
  filename = file.path(fig_dir, "breakpoints_afinn.png"),
  plot = p_af, width = 9, height = 5, dpi = 300
)

# ==========================================================
# 3. SAVE BREAKPOINT SUMMARY
# ==========================================================

summary_tbl <- tibble(
  sentiment_type = c("compound", "afinn"),
  n_breakpoints  = c(length(bp_comp_breaks), length(bp_af_breaks)),
  break_dates    = c(
    paste(as.character(bp_comp_breaks), collapse = "; "),
    paste(as.character(bp_af_breaks), collapse = "; ")
  )
)

write_csv(summary_tbl, here("reports", "structural_breaks_summary.csv"))

# ==========================================================
message("Structural break detection completed.")
# ==========================================================

# ==========================================================
# Script: 07_event_alignment.R
# Purpose:
#   Align sentiment with key Nigerian socio-political events,
#   using ± window_days around each event date.
#
#   - Builds daily sentiment around each event (AFINN + compound)
#   - Produces event-aligned plots
#   - Computes before/after differences per event
#
# Input:
#   data/processed/04_sentiment_tweets.rds
#
# Outputs:
#   data/processed/07_event_windows_daily.csv
#   reports/event_alignment_summary.csv
#   outputs/figures/event_alignment_compound.png
#   outputs/figures/event_alignment_afinn.png
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(lubridate)
library(ggplot2)
library(here)
library(readr)
library(purrr)
library(tidyr)
library(stringr)
library(tibble)

# ---- 1. Load sentiment data ----

sent_path <- here("data", "processed", "04_sentiment_tweets.rds")
if (!file.exists(sent_path)) {
  stop("04_sentiment_tweets.rds not found. Run 04_sentiment_scoring.R first.")
}

sentiment <- readRDS(sent_path)
message("Loaded sentiment dataset with ", nrow(sentiment), " tweets.")

# Ensure date column
if (!"date" %in% names(sentiment)) {
  sentiment <- sentiment %>%
    mutate(date = as_date(created_at))
}

min_date <- min(sentiment$date, na.rm = TRUE)
max_date <- max(sentiment$date, na.rm = TRUE)
message("Data span: ", as.character(min_date), " to ", as.character(max_date))

# ---- 2. Define key events (you can adjust dates if needed) ----

events <- tribble(
  ~event_id,       ~event_label,                                    ~event_date,
  "endsars",       "EndSARS protests (Oct 2020)",                   as.Date("2020-10-20"),
  "twitter_ban",   "Twitter ban (Jun 2021)",                        as.Date("2021-06-05"),
  "naira_crisis",  "Naira redesign / cash crisis (Feb 2023)",       as.Date("2023-02-01"),
  "election_2023", "General elections (Feb 2023)",                  as.Date("2023-02-25"),
  "subsidy_remov", "Fuel subsidy removal (May 2023)",               as.Date("2023-05-29")
)

# Keep only events that fall inside the data range
events <- events %>%
  filter(event_date >= min_date, event_date <= max_date)

if (nrow(events) == 0) {
  stop("No events fall within the data range. Check event dates or data coverage.")
}

message("Events used:")
print(events)

# ---- 3. Construct event windows ----

window_days <- 90L  # +/- 90 days around each event

build_event_window <- function(ev_row) {
  ev_date  <- ev_row$event_date
  ev_id    <- ev_row$event_id
  ev_label <- ev_row$event_label

  start_date <- ev_date - window_days
  end_date   <- ev_date + window_days

  df <- sentiment %>%
    filter(date >= start_date, date <= end_date) %>%
    group_by(date) %>%
    summarise(
      mean_compound = mean(compound, na.rm = TRUE),
      mean_afinn    = mean(afinn_sum, na.rm = TRUE),
      n_tweets      = n(),
      .groups = "drop"
    ) %>%
    mutate(
      event_id    = ev_id,
      event_label = ev_label,
      event_date  = ev_date,
      rel_day     = as.integer(date - ev_date)
    )

  df
}

event_windows <- events %>%
  split(.$event_id) %>%
  map_dfr(build_event_window)

# Save the raw daily window table
event_windows_path <- here("data", "processed", "07_event_windows_daily.csv")
write_csv(event_windows, event_windows_path)

message("Built event windows with ", nrow(event_windows), " daily observations.")

# ---- 4. Event-aligned plots ----

fig_dir <- here("outputs", "figures")
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

# 4.1 Compound sentiment around events
p_compound <- event_windows %>%
  ggplot(aes(x = rel_day, y = mean_compound)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = 0, linetype = "dotted") +
  geom_line() +
  facet_wrap(~ event_label, scales = "free_x") +
  labs(
    title = paste0("Event-aligned sentiment (sentimentr compound), ±", window_days, " days"),
    x = "Days relative to event (0 = event date)",
    y = "Mean compound sentiment"
  )

ggsave(
  filename = file.path(fig_dir, "event_alignment_compound.png"),
  plot = p_compound,
  width = 10,
  height = 6,
  dpi = 300
)

# 4.2 AFINN sentiment around events
p_afinn <- event_windows %>%
  ggplot(aes(x = rel_day, y = mean_afinn)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = 0, linetype = "dotted") +
  geom_line() +
  facet_wrap(~ event_label, scales = "free_x") +
  labs(
    title = paste0("Event-aligned sentiment (AFINN total), ±", window_days, " days"),
    x = "Days relative to event (0 = event date)",
    y = "Mean AFINN sentiment"
  )

ggsave(
  filename = file.path(fig_dir, "event_alignment_afinn.png"),
  plot = p_afinn,
  width = 10,
  height = 6,
  dpi = 300
)

# ---- 5. Before/after summary per event ----
# Use rel_day < 0 as 'before', rel_day > 0 as 'after'; exclude rel_day == 0

event_summary <- event_windows %>%
  filter(!is.na(mean_compound) | !is.na(mean_afinn)) %>%
  pivot_longer(
    cols = c(mean_compound, mean_afinn),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    period = case_when(
      rel_day < 0 ~ "before",
      rel_day > 0 ~ "after",
      TRUE        ~ "event_day"
    )
  ) %>%
  filter(period != "event_day") %>%
  group_by(event_id, event_label, metric, period) %>%
  summarise(
    mean_value = mean(value, na.rm = TRUE),
    n_days     = sum(!is.na(value)),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from  = period,
    values_from = c(mean_value, n_days),
    names_sep   = "_"
  ) %>%
  mutate(
    delta_after_minus_before = mean_value_after - mean_value_before
  )

# Optional: simple t-test on daily means, per event × metric
event_tests <- event_windows %>%
  filter(!is.na(mean_compound) | !is.na(mean_afinn)) %>%
  pivot_longer(
    cols = c(mean_compound, mean_afinn),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    period = case_when(
      rel_day < 0 ~ "before",
      rel_day > 0 ~ "after",
      TRUE        ~ "event_day"
    )
  ) %>%
  filter(period != "event_day") %>%
  group_by(event_id, event_label, metric) %>%
  summarise(
    mean_before = mean(value[period == "before"], na.rm = TRUE),
    mean_after  = mean(value[period == "after"], na.rm = TRUE),
    delta       = mean_after - mean_before,
    n_before    = sum(period == "before" & !is.na(value)),
    n_after     = sum(period == "after"  & !is.na(value)),
    p_value     = tryCatch(
      t.test(value[period == "before"], value[period == "after"])$p.value,
      error = function(e) NA_real_
    ),
    .groups = "drop"
  )

# Merge descriptive summary and t-test output
event_alignment_summary <- event_summary %>%
  left_join(
    event_tests %>%
      select(event_id, metric, p_value),
    by = c("event_id", "metric")
  )

summary_path <- here("reports", "event_alignment_summary.csv")
write_csv(event_alignment_summary, summary_path)

message("\n------ 07_event_alignment Summary ------")
print(event_alignment_summary)
message("---------------------------------------")
message("Daily windows saved to: ", event_windows_path)
message("Summary saved to: ", summary_path)
message("Figures saved to: ", fig_dir)
message("✅ 07_event_alignment.R completed successfully.")
# ==========================================================
# End of script
# ==========================================================

# ==========================================================
# Script: 09M_results_section.R
# Purpose:
#   Generate a draft academic Results section (markdown)
#   using all existing analyses and model performance files.
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(readr)
library(here)
library(glue)
library(stringr)
library(purrr)
library(tidyr)

message("=== 09M: Building Results section draft ===")

# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------

safe_read_csv <- function(path_rel) {
  path <- here(path_rel)
  if (!file.exists(path)) {
    message("Warning: file not found: ", path_rel)
    return(NULL)
  }
  readr::read_csv(path, show_col_types = FALSE)
}

extract_metric <- function(df, metric_pattern) {
  if (is.null(df) || nrow(df) == 0) return(NA_real_)

  out <- df %>%
    dplyr::filter(str_detect(metric, metric_pattern)) %>%
    dplyr::slice(1)

  if (nrow(out) == 0) return(NA_real_)

  suppressWarnings(as.numeric(out$value[1]))
}

fmt_pct <- function(x, digits = 1) {
  if (is.na(x)) return("NA")
  paste0(round(100 * x, digits), "%")
}

fmt_num <- function(x, digits = 3) {
  if (is.na(x)) return("NA")
  format(round(x, digits), nsmall = digits)
}

# ----------------------------------------------------------
# Load main sentiment file
# ----------------------------------------------------------

sent <- readRDS(here("data/processed/04_sentiment_tweets.rds"))

n_tweets <- nrow(sent)

# Compound stats always exist
compound_mean <- mean(sent$compound, na.rm = TRUE)
compound_sd   <- sd(sent$compound, na.rm = TRUE)

# AFINN stats only if column exists
if ("afinn" %in% names(sent)) {
  afinn_mean <- mean(sent$afinn, na.rm = TRUE)
  afinn_sd   <- sd(sent$afinn, na.rm = TRUE)
  has_afinn <- TRUE
} else {
  afinn_mean <- NA
  afinn_sd   <- NA
  has_afinn <- FALSE
  message("⚠ No 'afinn' column found. Skipping AFINN summaries.")
}

# Vader_class reconstruction
sent_classes <- sent %>%
  mutate(
    vader_class = case_when(
      compound >=  0.05 ~ "positive",
      compound <= -0.05 ~ "negative",
      TRUE              ~ "neutral"
    )
  )

vader_dist <- sent_classes %>%
  count(vader_class) %>%
  mutate(prop = n / sum(n))

neg_pct <- vader_dist$prop[vader_dist$vader_class == "negative"]
neu_pct <- vader_dist$prop[vader_dist$vader_class == "neutral"]
pos_pct <- vader_dist$prop[vader_dist$vader_class == "positive"]

share_negative <- mean(sent$compound < 0, na.rm = TRUE)

# ----------------------------------------------------------
# Model performance tables
# ----------------------------------------------------------

perf_09E <- safe_read_csv("reports/09E_model_performance_summary.csv")
perf_09F <- safe_read_csv("reports/09F_model_performance_summary.csv")
perf_09I_bin   <- safe_read_csv("reports/09I_char_bin_performance.csv")
perf_09I_multi <- safe_read_csv("reports/09I_char_multi_performance.csv")
perf_09J_xgb   <- safe_read_csv("reports/09J_xgb_performance_summary.csv")
perf_09J_hyb   <- safe_read_csv("reports/09J_xgb_hybrid_performance_summary.csv")
perf_09K_bin   <- safe_read_csv("reports/09K_word_bin_performance.csv")
perf_09K_multi <- safe_read_csv("reports/09K_word_multi_performance.csv")

model_summary <- tibble::tibble(
  model_id = c(
    "09E_logit_sentiment",
    "09F_logit_no_sentiment",
    "09I_char3_lasso_binary",
    "09I_char3_lasso_multi",
    "09K_word_lasso_binary",
    "09K_word_lasso_multi",
    "09J_xgb_numeric",
    "09J_xgb_hybrid"
  ),
  task = c(
    "binary", "binary", "binary", "multiclass",
    "binary", "multiclass", "binary", "binary"
  ),
  acc = c(
    extract_metric(perf_09E, "^accuracy$"),
    extract_metric(perf_09F, "^accuracy$"),
    extract_metric(perf_09I_bin, "^accuracy$"),
    extract_metric(perf_09I_multi, "^accuracy$"),
    extract_metric(perf_09K_bin, "^accuracy$"),
    extract_metric(perf_09K_multi, "^accuracy$"),
    extract_metric(perf_09J_xgb, "^accuracy$"),
    extract_metric(perf_09J_hyb, "^accuracy$")
  ),
  auc = c(
    extract_metric(perf_09E, "roc_auc"),
    extract_metric(perf_09F, "roc_auc"),
    extract_metric(perf_09I_bin, "roc_auc"),
    extract_metric(perf_09I_multi, "roc_auc"),
    extract_metric(perf_09K_bin, "roc_auc"),
    extract_metric(perf_09K_multi, "roc_auc"),
    extract_metric(perf_09J_xgb, "roc_auc"),
    extract_metric(perf_09J_hyb, "roc_auc")
  )
)

best_binary <- model_summary %>%
  filter(task == "binary") %>%
  arrange(desc(auc)) %>%
  dplyr::slice(1)

best_multi  <- model_summary %>%
  filter(task == "multiclass") %>%
  arrange(desc(auc)) %>%
  dplyr::slice(1)

# ----------------------------------------------------------
# Construct Results section text
# ----------------------------------------------------------

lines <- c()
lines <- c(lines, "# 4. Results", "")

# ---- 4.1 Sentiment landscape ----
lines <- c(lines, "## 4.1 Overall sentiment landscape", "")

compound_text <- glue(
  "Across {n_tweets} tweets, the mean VADER compound score was {fmt_num(compound_mean)} ",
  "(SD = {fmt_num(compound_sd)})."
)

if (has_afinn) {
  afinn_text <- glue(
    "AFINN scores showed a similar pattern (mean = {fmt_num(afinn_mean)}, SD = {fmt_num(afinn_sd)})."
  )
} else {
  afinn_text <- "AFINN scores were not available for this dataset."
}

sent_class_text <- glue(
  "{fmt_pct(neg_pct)} of tweets were classified as negative, ",
  "{fmt_pct(neu_pct)} as neutral, and {fmt_pct(pos_pct)} as positive."
)

lines <- c(lines, compound_text, afinn_text, "", sent_class_text, "")

# ---- 4.4 Predictive models ----
lines <- c(lines, "## 4.4 Predictive models", "")

lines <- c(
  lines,
  glue("The best binary classifier was **{best_binary$model_id}** (AUC = {fmt_num(best_binary$auc)})."),
  glue("The best multi-class model was **{best_multi$model_id}** (AUC = {fmt_num(best_multi$auc)})."),
  ""
)

# ----------------------------------------------------------
# Write output
# ----------------------------------------------------------

out_path <- here("reports/09M_results_section_draft.md")
writeLines(lines, out_path)

message("✅ 09M_results_section.R completed successfully.")
message("Draft saved to: ", out_path)

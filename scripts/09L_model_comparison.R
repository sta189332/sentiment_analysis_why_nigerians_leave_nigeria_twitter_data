# ==========================================================
# Script: 09L_model_comparison.R
# Purpose:
#   Compare performance of all sentiment prediction models:
#   - 09E: Logistic regression with sentiment features
#   - 09F: Logistic regression without sentiment features
#   - 09I: Char 3-gram TF-IDF + LASSO (binary + multiclass)
#   - 09J: XGBoost (numeric + hybrid)
#   - 09K: Word-level TF-IDF + LASSO (binary + multiclass)
#
# Inputs: various *performance*.csv files in reports/
#
# Outputs:
#   reports/09L_model_comparison_all.csv
#   reports/09L_model_comparison_binary.csv
#   reports/09L_model_comparison_multiclass.csv
#
#   outputs/figures/09L_binary_accuracy.png
#   outputs/figures/09L_binary_auc.png
#   outputs/figures/09L_multiclass_accuracy.png
#   outputs/figures/09L_multiclass_auc.png
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(readr)
library(tidyr)
library(purrr)
library(ggplot2)
library(here)
library(tibble)

# ----------------------------------------------------------
# Helper: robust reader for performance CSVs
# ----------------------------------------------------------
read_perf <- function(file_name,
                      model_id,
                      outcome_type,
                      model_family,
                      algorithm,
                      feature_set) {

  full_path <- here("reports", file_name)

  if (!file.exists(full_path)) {
    message("⚠️ Skipping ", model_id,
            " because file not found: ", full_path)
    return(NULL)
  }

  message("✅ Reading performance for ", model_id,
          " from: ", file_name)

  perf_raw <- readr::read_csv(full_path, show_col_types = FALSE)

  # Standardise to wide format if metric/value style
  if (all(c("metric", "value") %in% names(perf_raw))) {
    perf_wide <- perf_raw %>%
      tidyr::pivot_wider(names_from = metric, values_from = value)
  } else {
    perf_wide <- perf_raw
  }

  # Try to find accuracy column
  acc_candidates <- intersect(
    c("accuracy", "Accuracy", "acc"),
    names(perf_wide)
  )

  # Try to find ROC AUC column
  auc_candidates <- intersect(
    c("roc_auc",
      "roc_auc_binary",
      "roc_auc_macro_weighted",
      "roc_auc_macro",
      "AUC",
      "auc"),
    names(perf_wide)
  )

  accuracy <- if (length(acc_candidates) > 0) {
    perf_wide[[acc_candidates[1]]]
  } else {
    NA_real_
  }

  roc_auc <- if (length(auc_candidates) > 0) {
    perf_wide[[auc_candidates[1]]]
  } else {
    NA_real_
  }

  tibble::tibble(
    model_id     = model_id,
    model_family = model_family,
    algorithm    = algorithm,
    outcome_type = outcome_type,  # "binary" or "multiclass"
    feature_set  = feature_set,
    accuracy     = as.numeric(accuracy),
    roc_auc      = as.numeric(roc_auc)
  )
}

# ----------------------------------------------------------
# Define models and corresponding performance files
# ----------------------------------------------------------

models_meta <- tibble::tribble(
  ~model_id,                 ~file_name,                             ~outcome_type, ~model_family,          ~algorithm,                       ~feature_set,
  # 09E: logistic with sentiment features
  "09E_logit_sentiment",     "09E_model_performance_summary.csv",    "binary",      "logistic_regression", "Logistic regression",           "Sentiment + engagement + metadata",
  # 09F: logistic without sentiment
  "09F_logit_no_sentiment",  "09F_model_performance_summary.csv",    "binary",      "logistic_regression", "Logistic regression",           "Non-sentiment features only",
  # 09I: char 3-gram TF-IDF (binary)
  "09I_char3_lasso_binary",  "09I_char_bin_performance.csv",         "binary",      "lasso_logistic",      "L1-penalised logistic",         "Char 3-gram TF-IDF",
  # 09I: char 3-gram TF-IDF (multiclass)
  "09I_char3_lasso_multi",   "09I_char_multi_performance.csv",       "multiclass",  "lasso_multinomial",   "L1-penalised multinomial",      "Char 3-gram TF-IDF",
  # 09J: XGBoost numeric only (neg_flag)
  "09J_xgb_numeric",         "09J_xgb_performance_summary.csv",      "binary",      "xgboost",             "Gradient boosted trees",        "Numeric sentiment + covariates",
  # 09J: XGBoost hybrid (neg_flag + TF-IDF)
  "09J_xgb_hybrid",          "09J_xgb_hybrid_performance_summary.csv","binary",     "xgboost",             "Gradient boosted trees",        "Numeric + word TF-IDF hybrid",
  # 09K: word TF-IDF + LASSO (binary)  - assumes 09K writes this file
  "09K_word_lasso_binary",   "09K_word_bin_performance.csv",         "binary",      "lasso_logistic",      "L1-penalised logistic",         "Word TF-IDF (unigrams/bigrams)",
  # 09K: word TF-IDF + LASSO (multiclass) - assumes 09K writes this file
  "09K_word_lasso_multi",    "09K_word_multi_performance.csv",       "multiclass",  "lasso_multinomial",   "L1-penalised multinomial",      "Word TF-IDF (unigrams/bigrams)"
)

# ----------------------------------------------------------
# Read and combine all performance summaries
# ----------------------------------------------------------
message("---- Reading and combining model performance tables ----")

perf_all <- models_meta %>%
  pmap_dfr(~ read_perf(
    file_name    = ..2,
    model_id     = ..1,
    outcome_type = ..3,
    model_family = ..4,
    algorithm    = ..5,
    feature_set  = ..6
  ))

if (nrow(perf_all) == 0) {
  stop("❌ No performance files could be read. Check that 09E–09K scripts have been run.")
}

# Sort for readability: by outcome type, then ROC AUC desc, then accuracy
perf_all <- perf_all %>%
  arrange(outcome_type, desc(roc_auc), desc(accuracy))

# ----------------------------------------------------------
# Save comparison tables
# ----------------------------------------------------------
message("---- Saving comparison CSVs ----")

# All models
readr::write_csv(
  perf_all,
  here("reports", "09L_model_comparison_all.csv")
)

# Binary only
perf_binary <- perf_all %>%
  filter(outcome_type == "binary")

readr::write_csv(
  perf_binary,
  here("reports", "09L_model_comparison_binary.csv")
)

# Multiclass only
perf_multi <- perf_all %>%
  filter(outcome_type == "multiclass")

readr::write_csv(
  perf_multi,
  here("reports", "09L_model_comparison_multiclass.csv")
)

# ----------------------------------------------------------
# Visualisations: Accuracy and ROC AUC comparisons
# ----------------------------------------------------------
fig_dir <- here("outputs", "figures")
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

# Helper function: safe plot
plot_bar_metric <- function(df, metric, title, filename) {

  if (nrow(df) == 0 || all(is.na(df[[metric]]))) {
    message("⚠️ Skipping plot ", filename, " — no non-NA values for ", metric)
    return(invisible(NULL))
  }

  p <- df %>%
    mutate(model_id = factor(model_id,
                             levels = df %>% arrange(.data[[metric]]) %>% pull(model_id))) %>%
    ggplot(aes(x = model_id, y = .data[[metric]])) +
    geom_col() +
    coord_flip() +
    labs(
      title = title,
      x     = "Model",
      y     = metric
    ) +
    theme_minimal(base_size = 11)

  ggsave(
    filename = here("outputs", "figures", filename),
    plot     = p,
    width    = 7,
    height   = 4
  )

  invisible(NULL)
}

message("---- Creating comparison plots ----")

# Binary plots
plot_bar_metric(
  perf_binary,
  metric   = "accuracy",
  title    = "Binary models: Accuracy comparison",
  filename = "09L_binary_accuracy.png"
)

plot_bar_metric(
  perf_binary,
  metric   = "roc_auc",
  title    = "Binary models: ROC AUC comparison",
  filename = "09L_binary_auc.png"
)

# Multiclass plots
plot_bar_metric(
  perf_multi,
  metric   = "accuracy",
  title    = "Multiclass models: Accuracy comparison",
  filename = "09L_multiclass_accuracy.png"
)

plot_bar_metric(
  perf_multi,
  metric   = "roc_auc",
  title    = "Multiclass models: ROC AUC comparison",
  filename = "09L_multiclass_auc.png"
)

message("✅ 09L_model_comparison.R completed successfully.")
message("   - CSVs written to: reports/09L_model_comparison_*.csv")
message("   - Figures written to: outputs/figures/09L_*.png")

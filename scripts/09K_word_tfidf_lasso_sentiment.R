# ==========================================================
# Script: 09K_word_tfidf_lasso_sentiment.R
# Purpose:
#   Word-level TF-IDF + LASSO models for:
#     (1) Binary sentiment: neg_flag (non_negative vs negative)
#         - event class = "negative"
#     (2) Multi-class sentiment: vader_class
#         (negative / neutral / positive)
#
# Inputs:
#   data/processed/04_sentiment_tweets.rds
#
# Outputs:
#   Binary:
#     reports/09K_word_bin_performance.csv
#     reports/09K_word_bin_confusion_matrix.csv
#     reports/09K_word_bin_top50_terms.csv
#     data/processed/09K_word_bin_predictions.rds
#
#   Multi-class:
#     reports/09K_word_multi_performance.csv
#     reports/09K_word_multi_confusion_matrix.csv
#     reports/09K_word_multi_top50_terms.csv
#     data/processed/09K_word_multi_predictions.rds
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(readr)
library(here)
library(text2vec)
library(Matrix)
library(glmnet)
library(yardstick)
library(tibble)
library(purrr)

set.seed(20251129)

cat("=== Step 1: Load and prepare data ===\n")

sent <- readRDS(here("data/processed/04_sentiment_tweets.rds"))
cat("Rows in raw sentiment data:", nrow(sent), "\n")

data_base <- sent %>%
  mutate(
    text = tolower(as.character(text)),
    neg_flag = if_else(compound < 0, "negative", "non_negative"),
    vader_class = case_when(
      compound >=  0.05 ~ "positive",
      compound <= -0.05 ~ "negative",
      TRUE              ~ "neutral"
    )
  ) %>%
  filter(!is.na(text), text != "") %>%
  drop_na(compound, neg_flag, vader_class) %>%
  mutate(
    neg_flag = factor(neg_flag,
                      levels = c("non_negative", "negative")),
    vader_class = factor(vader_class,
                         levels = c("negative", "neutral", "positive"))
  )

cat("After cleaning:", nrow(data_base), "rows\n")
cat("neg_flag distribution:\n")
print(prop.table(table(data_base$neg_flag)))
cat("vader_class distribution:\n")
print(prop.table(table(data_base$vader_class)))

# ----------------------------------------------------------
# 2. Train / test split
# ----------------------------------------------------------
cat("=== Step 2: Train-test split ===\n")
idx <- sample(seq_len(nrow(data_base)), size = 0.8 * nrow(data_base))
train <- data_base[idx, ]
test  <- data_base[-idx, ]

cat("Train rows:", nrow(train), "\n")
cat("Test rows:",  nrow(test), "\n")

# ----------------------------------------------------------
# 3. Text2vec vocabulary + TF-IDF
# ----------------------------------------------------------
cat("=== Step 3: Build word-level TF-IDF ===\n")

it_train <- itoken(train$text, progressbar = FALSE)
vocab <- create_vocabulary(it_train)

cat("Initial vocabulary size:", nrow(vocab), "\n")

# Light pruning to remove ultra-rare words
vocab <- prune_vocabulary(
  vocab,
  term_count_min = 10,
  doc_proportion_min = 0.0001,
  doc_proportion_max = 0.5
)

cat("Pruned vocabulary size:", nrow(vocab), "\n")

vectorizer <- vocab_vectorizer(vocab)

dtm_train <- create_dtm(it_train, vectorizer)

it_test <- itoken(test$text, progressbar = FALSE)
dtm_test <- create_dtm(it_test, vectorizer)

cat("DTM train:", nrow(dtm_train), "x", ncol(dtm_train), "\n")
cat("DTM test :", nrow(dtm_test), "x", ncol(dtm_test), "\n")

tfidf <- TfIdf$new()
dtm_train_tfidf <- fit_transform(dtm_train, tfidf)
dtm_test_tfidf  <- transform(dtm_test, tfidf)

# ----------------------------------------------------------
# Helper: build coefficient tibble (single response)
# ----------------------------------------------------------
coef_to_tibble <- function(coef_obj, feature_names, response_name = NULL) {
  # coef_obj is a "dgCMatrix" for a single response
  idx <- which(coef_obj != 0)
  if (length(idx) == 0) {
    return(tibble(
      term = character(),
      coefficient = numeric(),
      abs_coefficient = numeric(),
      response = character()
    ))
  }

  # Drop intercept (row 1)
  intercept <- coef_obj[1, 1]
  coef_vec <- as.numeric(coef_obj[-1, 1])
  terms <- feature_names

  tibble(
    term = terms,
    coefficient = coef_vec,
    abs_coefficient = abs(coef_vec),
    response = if (is.null(response_name)) "binary" else response_name
  ) %>%
    arrange(desc(abs_coefficient))
}

# ----------------------------------------------------------
# 4. MODEL 1: Binary LASSO (neg_flag, event = "negative")
# ----------------------------------------------------------
cat("----- MODEL 1: BINARY (neg_flag, negative as event) -----\n")

y_bin <- if_else(train$neg_flag == "negative", 1, 0)

cv_bin <- cv.glmnet(
  x = dtm_train_tfidf,
  y = y_bin,
  family = "binomial",
  alpha = 1,
  nfolds = 5,
  type.measure = "auc"
)

lambda_bin <- cv_bin$lambda.min
cat("Chosen lambda (binary):", lambda_bin, "\n")

# Probabilities for class "negative" (event)
prob_neg <- as.numeric(
  predict(cv_bin$glmnet.fit,
          newx = dtm_test_tfidf,
          s = lambda_bin,
          type = "response")
)

pred_class_bin <- if_else(prob_neg >= 0.5, "negative", "non_negative") %>%
  factor(levels = levels(train$neg_flag))

results_bin <- tibble(
  status_id = test$status_id,
  truth = test$neg_flag,
  .pred_negative = prob_neg,
  .pred_non_negative = 1 - prob_neg,
  .pred_class = pred_class_bin
)

acc_bin <- accuracy(results_bin,
                    truth = truth,
                    estimate = .pred_class)

roc_bin <- roc_auc(
  results_bin,
  truth = truth,
  .pred_negative,
  event_level = "second"  # "negative" is the event
)

cat("Binary Accuracy =", round(acc_bin$.estimate, 4), "\n")
cat("Binary ROC AUC  =", round(roc_bin$.estimate, 4), "\n")

cm_bin <- conf_mat(results_bin,
                   truth = truth,
                   estimate = .pred_class)

# Coefficients / variable importance
coef_bin <- coef(cv_bin$glmnet.fit, s = lambda_bin)
vip_bin <- coef_to_tibble(coef_bin,
                          feature_names = colnames(dtm_train_tfidf)) %>%
  slice_head(n = 50)

# Save binary outputs
write_csv(
  tibble(
    metric = c("accuracy", "roc_auc"),
    value  = c(acc_bin$.estimate, roc_bin$.estimate)
  ),
  here("reports", "09K_word_bin_performance.csv")
)

write_csv(as_tibble(cm_bin$table),
          here("reports", "09K_word_bin_confusion_matrix.csv"))

write_csv(vip_bin,
          here("reports", "09K_word_bin_top50_terms.csv"))

saveRDS(results_bin,
        here("data/processed", "09K_word_bin_predictions.rds"))

# ----------------------------------------------------------
# 5. MODEL 2: Multinomial LASSO (vader_class)
# ----------------------------------------------------------
cat("----- MODEL 2: MULTI-CLASS (vader_class) -----\n")

y_multi <- train$vader_class

cv_multi <- cv.glmnet(
  x = dtm_train_tfidf,
  y = y_multi,
  family = "multinomial",
  alpha = 1,
  nfolds = 5,
  type.measure = "class"
)

lambda_multi <- cv_multi$lambda.min
cat("Chosen lambda (multi-class):", lambda_multi, "\n")

prob_multi_arr <- predict(
  cv_multi$glmnet.fit,
  newx = dtm_test_tfidf,
  s = lambda_multi,
  type = "response"
)

# prob_multi_arr: 3D array [n_obs, n_classes, 1]
prob_multi <- prob_multi_arr[, , 1]
prob_multi <- as.matrix(prob_multi)

# Ensure column order matches factor levels
cls <- levels(y_multi)
prob_multi <- prob_multi[, cls, drop = FALSE]

pred_class_multi <- factor(
  cls[apply(prob_multi, 1, which.max)],
  levels = cls
)

results_multi <- tibble(
  status_id = test$status_id,
  truth = test$vader_class,
  .pred_negative = prob_multi[, "negative"],
  .pred_neutral  = prob_multi[, "neutral"],
  .pred_positive = prob_multi[, "positive"],
  .pred_class    = pred_class_multi
)

acc_multi <- accuracy(results_multi,
                      truth = truth,
                      estimate = .pred_class)

roc_multi <- roc_auc(
  results_multi,
  truth = truth,
  .pred_negative, .pred_neutral, .pred_positive,
  estimator = "macro_weighted"
)

cat("Multi-class ACC =", round(acc_multi$.estimate, 4), "\n")
cat("Multi-class AUC =", round(roc_multi$.estimate, 4), "\n")

cm_multi <- conf_mat(results_multi,
                     truth = truth,
                     estimate = .pred_class)

# Coefficients / variable importance (aggregate across classes)
coef_multi_list <- coef(cv_multi$glmnet.fit, s = lambda_multi)

vip_multi <- map2_dfr(
  coef_multi_list,
  names(coef_multi_list),
  ~ coef_to_tibble(.x, feature_names = colnames(dtm_train_tfidf),
                   response_name = .y)
) %>%
  group_by(term) %>%
  summarise(
    max_abs_coef = max(abs_coefficient, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(max_abs_coef)) %>%
  slice_head(n = 50)

# Save multi-class outputs
write_csv(
  tibble(
    metric = c("accuracy", "roc_auc_macro_weighted"),
    value  = c(acc_multi$.estimate, roc_multi$.estimate)
  ),
  here("reports", "09K_word_multi_performance.csv")
)

write_csv(as_tibble(cm_multi$table),
          here("reports", "09K_word_multi_confusion_matrix.csv"))

write_csv(vip_multi,
          here("reports", "09K_word_multi_top50_terms.csv"))

saveRDS(results_multi,
        here("data/processed", "09K_word_multi_predictions.rds"))

cat("✅ 09K_word_tfidf_lasso_sentiment.R completed SUCCESSFULLY.\n")

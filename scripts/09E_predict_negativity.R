# ==========================================================
<<<<<<< HEAD
# Script: 09E_predict_negativity.R  (FINAL WORKING + SMOTE)
# Purpose: Predict negative tweets using tidy modelling.
=======
# Script: 09E_predict_negativity.R  (FINAL FIXED VERSION)
# Purpose:
#   Predict negative tweets using language, engagement,
#   timing, and sentiment features.
#
# Inputs:
#   data/processed/04_sentiment_tweets.rds
#   data/processed/02_tweets_clean.rds
#
# Outputs:
#   reports/09E_model_performance_summary.csv
#   reports/09E_model_coef_logit.csv
#   reports/09E_model_confusion_matrix.csv
#   data/processed/09E_predictions.rds
#   outputs/figures/09E_roc_curve_logit.png
>>>>>>> af0cf8c83fef3b47d786675005bc368b199f2e36
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

# ---- Libraries ----
library(dplyr)
library(tidyr)
library(readr)
library(lubridate)
library(forcats)
library(ggplot2)
library(here)

library(rsample)
library(recipes)
library(parsnip)
library(workflows)
library(yardstick)
library(tune)
library(dials)
library(vip)
library(purrr)
<<<<<<< HEAD
library(themis)   # <-- for SMOTE balancing

# ==========================================================
# 1. Load datasets
# ==========================================================

sent  <- readRDS(here("data/processed/04_sentiment_tweets.rds"))
clean <- readRDS(here("data/processed/02_tweets_clean.rds"))

message("Loaded sentiment dataset: ", nrow(sent))
message("Loaded clean dataset: ", nrow(clean))

# Ensure both have identical ID types
sent  <- sent  %>% mutate(status_id = as.double(status_id))
clean <- clean %>% mutate(status_id = as.double(status_id))

# ==========================================================
# 2. Create modelling-safe versions of logicals
# ==========================================================

clean_mod <- clean %>%
  mutate(
    mod_is_quote   = as.numeric(is_quote),
    mod_is_retweet = as.numeric(is_retweet)
  )

# Keep only modelling columns to avoid conflicts
clean_mod <- clean_mod %>%
  select(
    status_id,
    reply_count,
    retweet_count,
    favorite_count,
    user_followers_count,
    mod_is_quote,
    mod_is_retweet
  )

# ==========================================================
# 3. Merge modelling features (FINAL FIXED VERSION)
# ==========================================================

model_df <- sent %>%
  left_join(clean_mod, by = "status_id") %>%

  # ---- FIX DUPLICATE COLUMNS (.x and .y) ----
rename(
  reply_count        = reply_count.y,
  retweet_count      = retweet_count.y,
  favorite_count     = favorite_count.y,
  user_followers_count = user_followers_count.y
) %>%
  select(-reply_count.x,
         -retweet_count.x,
         -favorite_count.x,
         -user_followers_count.x) %>%
  # --------------------------------------------

mutate(
  lang_group = case_when(
    lang %in% c("en", "pcm") ~ lang,
    lang %in% c("und", "", NA) ~ "und",
    TRUE ~ "other"
  ),
  lang_group = factor(lang_group),
  hour      = hour(created_at),
  wday      = wday(created_at, label = TRUE, week_start = 1),
  month_num = month(created_at),
  neg_flag  = if_else(compound < 0, "negative", "non_negative")
) %>%
  filter(!is.na(neg_flag)) %>%
  drop_na()

# ==========================================================
# 4. Select modelling-safe columns
# ==========================================================

model_df <- model_df %>%
  select(
    neg_flag,
    lang_group,
    hour, wday, month_num,
    compound, afinn_sum,
    reply_count, retweet_count, favorite_count,
    user_followers_count,
    mod_is_quote, mod_is_retweet
  )

# ==========================================================
# 5. Train-test split
# ==========================================================

set.seed(20251128)
split <- initial_split(model_df, prop = 0.80, strata = neg_flag)
train_data <- training(split)
test_data  <- testing(split)

# ==========================================================
# 6. Preprocessing recipe (WITH SMOTE)
# ==========================================================

rec <- recipe(neg_flag ~ ., data = train_data) %>%
  step_dummy(lang_group, wday) %>%
  step_zv(all_predictors()) %>%
  step_log(
    reply_count,
    retweet_count,
    favorite_count,
    user_followers_count,
    offset = 1
  ) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(neg_flag)   # <-- FIX: handle imbalance

# ==========================================================
# 7. Logistic regression (LASSO)
# ==========================================================

logit_mod <- logistic_reg(
  penalty = tune(),
  mixture = 1
) %>%
  set_engine("glmnet") %>%
=======

# ---- 1. Load datasets ----
sent_path  <- here("data", "processed", "04_sentiment_tweets.rds")
clean_path <- here("data", "processed", "02_tweets_clean.rds")

sent  <- readRDS(sent_path)
clean <- readRDS(clean_path)

message("Loaded sentiment dataset: ", nrow(sent))
message("Loaded clean tweets: ", nrow(clean))

# ---- 2. Merge engagement metadata into sentiment ----
model_df <- sent %>%
  left_join(
    clean %>%
      select(
        status_id,
        reply_count,
        retweet_count,
        favorite_count,
        user_followers_count,
        is_retweet,
        is_quote
      ),
    by = "status_id"
  )

# ---- 3. Create modelling variables ----
model_df <- model_df %>%
  mutate(
    lang_group = case_when(
      lang %in% c("en", "pcm") ~ lang,
      lang %in% c("und", "", NA) ~ "und",
      TRUE ~ "other"
    ),
    lang_group = factor(lang_group),

    hour = hour(created_at),
    wday = wday(created_at, label = TRUE, week_start = 1),
    month_num = month(created_at),

    neg_flag = if_else(compound < 0, "negative", "non_negative")
  ) %>%
  filter(!is.na(neg_flag)) %>%
  drop_na()

model_df$neg_flag <- factor(
  model_df$neg_flag,
  levels = c("non_negative", "negative")
)

message("Final modelling DF size: ", nrow(model_df))

# ---- 4. Train-test split ----
set.seed(20251128)
split <- initial_split(model_df, prop = 0.80, strata = neg_flag)

train_data <- training(split)
test_data  <- testing(split)

# ---- 5. Preprocessing recipe ----
rec <- recipe(neg_flag ~ ., data = train_data) %>%
  update_role(status_id, new_role = "ID") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_log(
    user_followers_count,
    reply_count,
    retweet_count,
    favorite_count,
    offset = 1
  ) %>%
  step_normalize(all_numeric_predictors())

# ---- 6. Logistic regression with L1 penalty (LASSO) ----
logit_mod <- logistic_reg(
  penalty = tune(),
  mixture = 1
) %>% set_engine("glmnet") %>%
>>>>>>> af0cf8c83fef3b47d786675005bc368b199f2e36
  set_mode("classification")

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(logit_mod)

set.seed(20251128)
cv <- vfold_cv(train_data, v = 5, strata = neg_flag)

grid <- grid_regular(
  penalty(range = c(-4, 0)),
  levels = 10
)

tuned <- tune_grid(
  wf,
  resamples = cv,
  grid = grid,
  metrics = metric_set(roc_auc, accuracy)
)

<<<<<<< HEAD
best <- select_best(tuned, metric = "roc_auc")
final_wf <- finalize_workflow(wf, best)

# ==========================================================
# 8. Fit final model
# ==========================================================

final_fit <- final_wf %>% fit(train_data)

# ==========================================================
# 9. Predict
# ==========================================================

# ---- 9. Predict ----
preds <- final_fit %>%
  predict(test_data, type = "prob") %>%
  bind_cols(predict(final_fit, test_data)) %>%
  bind_cols(test_data %>% select(neg_flag)) %>%
  rename(pred_class = .pred_class) %>%
  mutate(
    neg_flag = factor(neg_flag, levels = c("non_negative", "negative"))
  )

# ==========================================================
# 10. Metrics  (FINAL FIX)
# ==========================================================

# Ensure same factor level order
preds <- preds %>%
  mutate(
    neg_flag   = factor(neg_flag,   levels = c("non_negative", "negative")),
    pred_class = factor(pred_class, levels = c("non_negative", "negative"))
  )

# Use correct probability for the "negative" class
roc_obj <- roc_auc(preds, truth = neg_flag, .pred_negative, event_level = "second")

acc_obj <- accuracy(preds, truth = neg_flag, pred_class)

cm <- conf_mat(preds, truth = neg_flag, pred_class)

# ==========================================================
# 11. Save outputs
# ==========================================================

write_csv(
  tibble(metric = c("roc_auc", "accuracy"),
         value = c(roc_obj$.estimate, acc_obj$.estimate)),
  here("reports/09E_model_performance_summary.csv")
)

write_csv(vip_tbl, here("reports/09E_model_coef_logit.csv"))
write_csv(as_tibble(cm$table),
          here("reports/09E_model_confusion_matrix.csv"))

saveRDS(preds, here("data/processed/09E_predictions.rds"))

# ==========================================================
# 12. ROC curve
# ==========================================================

roc_df <- roc_curve(preds, truth = neg_flag, .pred_negative)

p <- ggplot(roc_df, aes(1 - specificity, sensitivity)) +
  geom_path(color = "blue", linewidth = 1) +
=======
best <- select_best(tuned, "roc_auc")
final_wf <- finalize_workflow(wf, best)

# ---- 7. Fit model ----
final_fit <- final_wf %>% fit(data = train_data)

# ---- 8. Predict on test data ----
preds <- final_fit %>%
  predict(test_data, type = "prob") %>%
  bind_cols(predict(final_fit, test_data)) %>%
  bind_cols(test_data %>% select(status_id, neg_flag))

colnames(preds)[colnames(preds) == ".pred_class"] <- "pred_class"

# ---- 9. Compute evaluation metrics ----
roc_obj <- roc_auc(preds, truth = neg_flag, .pred_negative)
acc_obj <- accuracy(preds, truth = neg_flag, pred_class)
cm      <- conf_mat(preds, truth = neg_flag, pred_class)

# ---- 10. Variable importance ----
vip_tbl <- final_fit %>%
  extract_fit_parsnip() %>%
  vip::vi(lambda = best$penalty) %>%
  arrange(desc(Importance))

# ---- 11. Save outputs ----
write_csv(
  tibble(
    metric = c("roc_auc", "accuracy"),
    value = c(roc_obj$.estimate, acc_obj$.estimate)
  ),
  here("reports", "09E_model_performance_summary.csv")
)

write_csv(vip_tbl, here("reports", "09E_model_coef_logit.csv"))
write_csv(as_tibble(cm$table),
          here("reports", "09E_model_confusion_matrix.csv"))

saveRDS(preds, here("data", "processed", "09E_predictions.rds"))

# ---- 12. ROC curve ----
roc_df <- roc_curve(preds, truth = neg_flag, .pred_negative)

p <- ggplot(roc_df, aes(1 - specificity, sensitivity)) +
  geom_path(color = "blue", size = 1) +
>>>>>>> af0cf8c83fef3b47d786675005bc368b199f2e36
  geom_abline(linetype = "dashed") +
  coord_equal() +
  theme_minimal() +
  labs(
    title = "ROC Curve – Predicting Negative Tweets",
    x = "1 - Specificity",
    y = "Sensitivity"
  )

ggsave(
  here("outputs/figures/09E_roc_curve_logit.png"),
  p, width = 6, height = 6, dpi = 300
)

message("ROC AUC: ", roc_obj$.estimate)
message("Accuracy: ", acc_obj$.estimate)
message("Model complete.")
<<<<<<< HEAD
message("✅ 09E_predict_negativity.R completed successfully.")
=======

message("✅ 09E_predict_negativity.R completed successfully.")
# ==========================================================
# End of script
# ==========================================================
>>>>>>> af0cf8c83fef3b47d786675005bc368b199f2e36

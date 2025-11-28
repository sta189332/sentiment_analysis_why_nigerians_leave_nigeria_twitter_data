# ==========================================================
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

message("✅ 09E_predict_negativity.R completed successfully.")
# ==========================================================
# End of script
# ==========================================================

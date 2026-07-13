# ==========================================================
# Script: 04A_validate_sentiment.R
# Purpose: Validate AFINN and VADER classifications against
# independently coded and adjudicated human labels.
# ==========================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(here)
  library(tibble)
})

set.seed(20260707)

validation_n <- 500L
classes <- c("negative", "neutral", "positive")
binary_classes <- c("negative", "non_negative")

sentiment_path <- here("data", "processed", "04_sentiment_tweets.rds")
clean_path <- here("data", "processed", "02_tweets_clean.rds")
validation_dir <- here("reports", "validation")
dir.create(validation_dir, recursive = TRUE, showWarnings = FALSE)

coder1_path <- here(validation_dir, "coder_1_labels.csv")
coder2_path <- here(validation_dir, "coder_2_labels.csv")
adjudication_path <- here(validation_dir, "adjudication_labels.csv")

safe_ratio <- function(x, y) if (is.na(y) || y == 0) NA_real_ else x / y

kappa_levels <- function(x, y, levels) {
  tab <- table(factor(x, levels = levels), factor(y, levels = levels))
  n <- sum(tab)
  if (!n) return(NA_real_)
  po <- sum(diag(tab)) / n
  pe <- sum(rowSums(tab) * colSums(tab)) / n^2
  if (abs(1 - pe) < .Machine$double.eps) return(NA_real_)
  (po - pe) / (1 - pe)
}

confusion_metrics <- function(truth, estimate, levels = classes) {
  tab <- table(
    truth = factor(truth, levels = levels),
    predicted = factor(estimate, levels = levels)
  )

  class_metrics <- lapply(levels, function(level) {
    tp <- tab[level, level]
    fp <- sum(tab[, level]) - tp
    fn <- sum(tab[level, ]) - tp
    precision <- safe_ratio(tp, tp + fp)
    recall <- safe_ratio(tp, tp + fn)
    f1 <- if (is.na(precision) || is.na(recall) || precision + recall == 0) {
      NA_real_
    } else {
      2 * precision * recall / (precision + recall)
    }

    tibble(
      class = level,
      support = sum(tab[level, ]),
      predicted_n = sum(tab[, level]),
      TP = tp,
      FP = fp,
      FN = fn,
      precision = precision,
      recall = recall,
      F1 = f1
    )
  }) %>% bind_rows()

  complete_reference <- all(rowSums(tab) > 0)

  overall <- tibble(
    accuracy = sum(diag(tab)) / sum(tab),
    n_reference_classes = sum(rowSums(tab) > 0),
    complete_three_class_reference = complete_reference,
    macro_precision = if (complete_reference) mean(class_metrics$precision) else NA_real_,
    macro_recall = if (complete_reference) mean(class_metrics$recall) else NA_real_,
    macro_f1 = if (complete_reference) mean(class_metrics$F1) else NA_real_,
    cohen_kappa = kappa_levels(truth, estimate, levels)
  )

  list(confusion = tab, by_class = class_metrics, overall = overall)
}

binary_metrics <- function(truth, estimate) {
  truth <- if_else(truth == "negative", "negative", "non_negative")
  estimate <- if_else(estimate == "negative", "negative", "non_negative")

  tab <- table(
    truth = factor(truth, levels = binary_classes),
    predicted = factor(estimate, levels = binary_classes)
  )

  tp <- tab["negative", "negative"]
  fp <- tab["non_negative", "negative"]
  fn <- tab["negative", "non_negative"]
  tn <- tab["non_negative", "non_negative"]
  precision <- safe_ratio(tp, tp + fp)
  recall <- safe_ratio(tp, tp + fn)
  specificity <- safe_ratio(tn, tn + fp)
  f1 <- if (is.na(precision) || is.na(recall) || precision + recall == 0) {
    NA_real_
  } else {
    2 * precision * recall / (precision + recall)
  }

  majority_baseline <- max(prop.table(table(truth)))
  accuracy <- (tp + tn) / sum(tab)

  list(
    confusion = tab,
    overall = tibble(
      accuracy = accuracy,
      majority_class = names(which.max(table(truth))),
      majority_baseline_accuracy = as.numeric(majority_baseline),
      accuracy_above_majority_baseline = accuracy - majority_baseline,
      negative_precision = precision,
      negative_recall = recall,
      negative_f1 = f1,
      specificity = specificity,
      balanced_accuracy = mean(c(recall, specificity), na.rm = TRUE),
      cohen_kappa = kappa_levels(truth, estimate, binary_classes),
      TP = tp,
      FP = fp,
      FN = fn,
      TN = tn
    )
  )
}

if (!file.exists(sentiment_path)) stop("Run 04_sentiment_scoring.R first.")
sent <- readRDS(sentiment_path)

required <- c("status_id", "compound", "afinn_sum")
missing <- setdiff(required, names(sent))
if (length(missing)) stop("Missing sentiment columns: ", paste(missing, collapse = ", "))

sent <- sent %>% mutate(status_id = as.character(status_id))

if (!"text" %in% names(sent)) {
  if (!file.exists(clean_path)) stop("Tweet text is unavailable.")
  clean <- readRDS(clean_path) %>%
    transmute(status_id = as.character(status_id), text) %>%
    distinct(status_id, .keep_all = TRUE)
  sent <- left_join(sent, clean, by = "status_id")
}

if (anyDuplicated(sent$status_id)) stop("Duplicate status IDs in sentiment data.")
if (nrow(sent) < validation_n) stop("Fewer than 500 unique tweets are available.")

sample_ids <- sample(sent$status_id, validation_n, replace = FALSE)
validation_sample <- sent %>%
  filter(status_id %in% sample_ids) %>%
  slice(match(sample_ids, status_id)) %>%
  transmute(status_id, text)

blank_coder_file <- validation_sample %>% mutate(human_label = "")

if (!file.exists(coder1_path) && !file.exists(coder2_path)) {
  write_csv(blank_coder_file, coder1_path)
  write_csv(blank_coder_file, coder2_path)
  stop("Coder files created. Complete both human_label columns and rerun.")
}

if (!file.exists(coder1_path) || !file.exists(coder2_path)) {
  stop("Both coder files are required.")
}

coder1 <- read_csv(coder1_path, show_col_types = FALSE)
coder2 <- read_csv(coder2_path, show_col_types = FALSE)

check_coder <- function(x, name) {
  required <- c("status_id", "text", "human_label")
  if (!all(required %in% names(x))) stop(name, " has an invalid schema.")
  if (nrow(x) != validation_n) stop(name, " must contain 500 rows.")
  if (anyDuplicated(x$status_id)) stop(name, " contains duplicate IDs.")
  if (!identical(as.character(x$status_id), validation_sample$status_id)) stop(name, " IDs or row order changed.")
  if (!identical(as.character(x$text), validation_sample$text)) stop(name, " tweet text changed.")
  labels <- trimws(tolower(as.character(x$human_label)))
  if (any(!labels %in% classes)) stop(name, " contains blank or invalid labels.")
  labels
}

coder1_label <- check_coder(coder1, "coder_1_labels.csv")
coder2_label <- check_coder(coder2, "coder_2_labels.csv")

coder_comparison <- validation_sample %>%
  mutate(coder1_label = coder1_label, coder2_label = coder2_label)

disagreements <- coder_comparison %>% filter(coder1_label != coder2_label)

intercoder <- tibble(
  n_tweets = validation_n,
  n_agreements = sum(coder1_label == coder2_label),
  n_disagreements = nrow(disagreements),
  agreement_rate = mean(coder1_label == coder2_label),
  cohen_kappa = kappa_levels(coder1_label, coder2_label, classes)
)

write_csv(intercoder, here(validation_dir, "intercoder_agreement.csv"))

if (nrow(disagreements) > 0 && !file.exists(adjudication_path)) {
  write_csv(disagreements %>% mutate(adjudicated_label = ""), adjudication_path)
  stop("Adjudication file created. Complete adjudicated_label and rerun.")
}

if (nrow(disagreements) > 0) {
  adjudication <- read_csv(adjudication_path, show_col_types = FALSE)
  required_adj <- c("status_id", "text", "coder1_label", "coder2_label", "adjudicated_label")
  if (!all(required_adj %in% names(adjudication))) stop("Invalid adjudication schema.")
  if (!identical(as.character(adjudication$status_id), disagreements$status_id)) stop("Adjudication IDs or row order changed.")
  if (!identical(as.character(adjudication$text), disagreements$text)) stop("Adjudication text changed.")
  adj <- trimws(tolower(as.character(adjudication$adjudicated_label)))
  if (any(!adj %in% classes)) stop("Blank or invalid adjudicated labels.")
} else {
  adjudication <- disagreements
  adj <- character()
}

reference <- coder_comparison %>%
  mutate(
    final_human_label = if_else(coder1_label == coder2_label, coder1_label, NA_character_)
  )

if (nrow(disagreements)) {
  reference$final_human_label[match(adjudication$status_id, reference$status_id)] <- adj
}

validation <- sent %>%
  filter(status_id %in% reference$status_id) %>%
  slice(match(reference$status_id, status_id)) %>%
  transmute(
    status_id,
    text,
    compound,
    afinn_sum,
    vader_label = case_when(
      compound <= -0.05 ~ "negative",
      compound >= 0.05 ~ "positive",
      TRUE ~ "neutral"
    ),
    afinn_label = case_when(
      coalesce(afinn_sum, 0) < 0 ~ "negative",
      coalesce(afinn_sum, 0) > 0 ~ "positive",
      TRUE ~ "neutral"
    )
  ) %>%
  bind_cols(reference %>% select(coder1_label, coder2_label, final_human_label))

vader_three <- confusion_metrics(validation$final_human_label, validation$vader_label)
afinn_three <- confusion_metrics(validation$final_human_label, validation$afinn_label)
vader_binary <- binary_metrics(validation$final_human_label, validation$vader_label)
afinn_binary <- binary_metrics(validation$final_human_label, validation$afinn_label)

overall <- bind_rows(
  vader_three$overall %>% mutate(method = "VADER", .before = 1),
  afinn_three$overall %>% mutate(method = "AFINN", .before = 1)
)

by_class <- bind_rows(
  vader_three$by_class %>% mutate(method = "VADER", .before = 1),
  afinn_three$by_class %>% mutate(method = "AFINN", .before = 1)
)

binary_overall <- bind_rows(
  vader_binary$overall %>% mutate(method = "VADER", .before = 1),
  afinn_binary$overall %>% mutate(method = "AFINN", .before = 1)
)

reference_distribution <- validation %>%
  count(final_human_label, name = "n") %>%
  mutate(proportion = n / sum(n))

write_csv(overall, here(validation_dir, "sentiment_validation_overall.csv"))
write_csv(by_class, here(validation_dir, "sentiment_validation_by_class.csv"))
write_csv(binary_overall, here(validation_dir, "sentiment_validation_binary_overall.csv"))
write_csv(as.data.frame.matrix(vader_three$confusion) %>% rownames_to_column("truth"), here(validation_dir, "vader_confusion_matrix.csv"))
write_csv(as.data.frame.matrix(afinn_three$confusion) %>% rownames_to_column("truth"), here(validation_dir, "afinn_confusion_matrix.csv"))
write_csv(reference_distribution, here(validation_dir, "reference_label_distribution.csv"))
write_csv(validation, here(validation_dir, "sentiment_validation_audit.csv"))

print(intercoder, width = Inf)
print(overall, width = Inf)
print(binary_overall, width = Inf)
message("04A_validate_sentiment.R completed successfully.")

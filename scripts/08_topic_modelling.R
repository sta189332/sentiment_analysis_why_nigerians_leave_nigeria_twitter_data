# ==========================================================
# Script: 08_topic_modelling.R
# Purpose:
#   Build topic models on the 2018–2025 Twitter corpus using:
#     (1) LDA via topicmodels
#     (2) STM (if stm + quanteda are available)
#
# Inputs:
#   data/processed/03_tokens.rds
#   data/processed/02_tweets_clean.rds   (for metadata)
#
# Outputs:
#   data/processed/08_lda_doc_topics.rds
#   data/processed/08_lda_top_terms.rds
#   outputs/figures/lda_top_terms.png
#
#   (if STM available)
#   data/processed/08_stm_doc_topics.rds
#   data/processed/08_stm_top_terms.rds
#   outputs/figures/stm_top_terms.png
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(tidyr)
library(tidytext)
library(here)
library(readr)
library(ggplot2)
library(stringr)

# ----------------------------------------------------------
# 0. Parameters
# ----------------------------------------------------------
K_topics      <- 10        # number of topics for LDA / STM
min_term_df   <- 30        # minimum document frequency for a term
max_term_prop <- 0.50      # drop terms appearing in more than this share of docs

set.seed(1234)

# ----------------------------------------------------------
# 1. Load tokens and tweet-level metadata
# ----------------------------------------------------------

tokens_path <- here("data", "processed", "03_tokens.rds")
tweets_path <- here("data", "processed", "02_tweets_clean.rds")

if (!file.exists(tokens_path)) {
  stop("03_tokens.rds not found. Run 03_tokenize_data.R first.")
}
if (!file.exists(tweets_path)) {
  stop("02_tweets_clean.rds not found. Run 02_clean_data.R first.")
}

tokens <- readRDS(tokens_path)
tweets <- readRDS(tweets_path)

message("Loaded tokens: ", nrow(tokens), " rows.")
message("Loaded cleaned tweets: ", nrow(tweets), " rows.")

# Expect at least: status_id, word in tokens
if (!all(c("status_id", "word") %in% names(tokens))) {
  stop("tokens object must contain columns 'status_id' and 'word'.")
}

n_docs <- n_distinct(tokens$status_id)
message("Number of documents (unique status_id): ", n_docs)

# ----------------------------------------------------------
# 2. Build document–term counts and filter vocabulary
# ----------------------------------------------------------

# Basic document–term counts
doc_term_counts <- tokens %>%
  filter(!is.na(word), word != "") %>%
  count(status_id, word, name = "n")

# Term document frequency
term_docfreq <- doc_term_counts %>%
  group_by(word) %>%
  summarise(
    docfreq = n(),                       # number of documents containing the term
    .groups = "drop"
  ) %>%
  mutate(
    docfreq_prop = docfreq / n_docs
  )

# Filter vocabulary by frequency thresholds
term_keep <- term_docfreq %>%
  filter(
    docfreq >= min_term_df,
    docfreq_prop <= max_term_prop
  ) %>%
  arrange(desc(docfreq))

message("Vocabulary size after filtering: ", nrow(term_keep), " terms.")

if (nrow(term_keep) < K_topics * 5) {
  warning("Filtered vocabulary is small relative to number of topics.
          Consider reducing K_topics or lowering min_term_df.")
}

doc_term_filtered <- doc_term_counts %>%
  inner_join(term_keep %>% select(word), by = "word")

# ----------------------------------------------------------
# 3. Construct DTM for LDA
# ----------------------------------------------------------

message("Constructing DTM for LDA...")

dtm <- doc_term_filtered %>%
  tidytext::cast_dtm(document = status_id, term = word, value = n)

message("DTM has ", dtm$nrow, " documents and ", dtm$ncol, " terms.")

# ----------------------------------------------------------
# 4. LDA topic model
# ----------------------------------------------------------

library(topicmodels)

message("Fitting LDA with K = ", K_topics, " topics...")

lda_model <- topicmodels::LDA(
  dtm,
  k       = K_topics,
  control = list(seed = 1234)
)

# 4.1 Topic–term probabilities (beta)
lda_terms <- tidytext::tidy(lda_model, matrix = "beta") %>%
  # columns: topic, term, beta
  group_by(topic) %>%
  arrange(desc(beta), .by_group = TRUE)

# 4.2 Document–topic probabilities (gamma)
lda_doc_topics <- tidytext::tidy(lda_model, matrix = "gamma") %>%
  # columns: document, topic, gamma
  rename(
    status_id = document
  ) %>%
  mutate(
    status_id = as.character(status_id),
    topic     = as.integer(topic)
  )

# Save LDA outputs
lda_terms_path       <- here("data", "processed", "08_lda_top_terms.rds")
lda_doc_topics_path  <- here("data", "processed", "08_lda_doc_topics.rds")

saveRDS(lda_terms, lda_terms_path)
saveRDS(lda_doc_topics, lda_doc_topics_path)

message("Saved LDA top terms to: ", lda_terms_path)
message("Saved LDA doc–topic weights to: ", lda_doc_topics_path)

# ----------------------------------------------------------
# 5. Plot LDA top terms per topic
# ----------------------------------------------------------

message("Building LDA top-terms plot...")

top_n_terms <- 8

lda_top_terms_plot <- lda_terms %>%
  group_by(topic) %>%
  slice_max(order_by = beta, n = top_n_terms, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(
    term = reorder_within(term, beta, topic)
  ) %>%
  ggplot(aes(x = term, y = beta)) +
  geom_col() +
  coord_flip() +
  scale_x_reordered() +
  facet_wrap(~ topic, scales = "free_y") +
  labs(
    title = paste0("LDA top terms by topic (K = ", K_topics, ")"),
    x = "Term",
    y = "β (topic–term probability)"
  )

fig_dir <- here("outputs", "figures")
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

lda_fig_path <- file.path(fig_dir, "lda_top_terms.png")

ggsave(
  filename = lda_fig_path,
  plot     = lda_top_terms_plot,
  width    = 10,
  height   = 7,
  dpi      = 300
)

message("Saved LDA top-terms figure to: ", lda_fig_path)

# ----------------------------------------------------------
# 6. Optional: STM model (if stm + quanteda are installed)
# ----------------------------------------------------------

stm_available <- requireNamespace("stm", quietly = TRUE) &&
  requireNamespace("quanteda", quietly = TRUE) &&
  requireNamespace("quanteda.textmodels", quietly = TRUE)

if (!stm_available) {
  message("STM/quanteda not available. Skipping STM model.")
} else {

  message("STM and quanteda detected. Fitting STM model with K = ", K_topics, ".")

  library(quanteda)
  library(quanteda.textmodels)
  library(stm)

  # 6.1 Build per-document text from tokens for STM
  doc_text <- tokens %>%
    semi_join(term_keep, by = "word") %>%      # only keep vocabulary used above
    group_by(status_id) %>%
    summarise(
      text = paste(word, collapse = " "),
      .groups = "drop"
    )

  # Ensure alignment with tweets metadata
  meta <- tweets %>%
    mutate(status_id = as.character(status_id)) %>%
    select(status_id, created_at, lang, user_followers_count, reply_count,
           retweet_count, favorite_count, year, month, week)

  doc_text <- doc_text %>%
    mutate(status_id = as.character(status_id)) %>%
    inner_join(meta, by = "status_id")

  # 6.2 Build quanteda corpus and dfm
  corp <- corpus(doc_text, text_field = "text")
  toks <- tokens(corp)

  dfm_stm <- dfm(toks) %>%
    dfm_trim(
      min_docfreq = min_term_df,
      docfreq_type = "count"
    )

  # Ensure dfm aligns with meta
  meta_stm <- docvars(dfm_stm)

  # 6.3 Convert to STM input
  stm_input <- quanteda::convert(dfm_stm, to = "stm")

  # 6.4 Fit STM
  stm_model <- stm::stm(
    documents = stm_input$documents,
    vocab     = stm_input$vocab,
    data      = stm_input$meta,
    K         = K_topics,
    init.type = "Spectral",
    seed      = 1234
  )

  # 6.5 STM topic–term probabilities
  stm_terms <- tidytext::tidy(stm_model) %>%
    group_by(topic) %>%
    arrange(desc(beta), .by_group = TRUE)

  # 6.6 STM document–topic weights
  stm_doc_topics <- tidytext::tidy(stm_model, matrix = "gamma") %>%
    rename(
      status_id = document
    ) %>%
    mutate(
      status_id = as.character(status_id),
      topic     = as.integer(topic)
    )

  # Save STM outputs
  stm_terms_path      <- here("data", "processed", "08_stm_top_terms.rds")
  stm_doc_topics_path <- here("data", "processed", "08_stm_doc_topics.rds")

  saveRDS(stm_terms, stm_terms_path)
  saveRDS(stm_doc_topics, stm_doc_topics_path)

  message("Saved STM top terms to: ", stm_terms_path)
  message("Saved STM doc–topic weights to: ", stm_doc_topics_path)

  # 6.7 Plot STM top terms
  stm_top_terms_plot <- stm_terms %>%
    group_by(topic) %>%
    slice_max(order_by = beta, n = top_n_terms, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(
      term = reorder_within(term, beta, topic)
    ) %>%
    ggplot(aes(x = term, y = beta)) +
    geom_col() +
    coord_flip() +
    scale_x_reordered() +
    facet_wrap(~ topic, scales = "free_y") +
    labs(
      title = paste0("STM top terms by topic (K = ", K_topics, ")"),
      x = "Term",
      y = "β (topic–term probability)"
    )

  stm_fig_path <- file.path(fig_dir, "stm_top_terms.png")

  ggsave(
    filename = stm_fig_path,
    plot     = stm_top_terms_plot,
    width    = 10,
    height   = 7,
    dpi      = 300
  )

  message("Saved STM top-terms figure to: ", stm_fig_path)
}

# ----------------------------------------------------------
# 7. Console summary
# ----------------------------------------------------------

message("\n------ 08_topic_modelling Summary ------")
message("Documents (tweets): ", n_docs)
message("Filtered vocabulary size: ", nrow(term_keep))
message("LDA topics: ", K_topics)
if (stm_available) {
  message("STM topics: ", K_topics, " (STM successfully fitted).")
} else {
  message("STM was skipped due to missing packages.")
}
message("Figures directory: ", fig_dir)
message("✅ 08_topic_modelling.R completed successfully.")
# ==========================================================
# End of script
# ==========================================================

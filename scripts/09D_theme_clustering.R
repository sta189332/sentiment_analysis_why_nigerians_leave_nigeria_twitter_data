# ==========================================================
# Script: 09D_theme_clustering.R
# Purpose:
#   Unsupervised clustering of themes using quanteda, TF-IDF,
#   k-means, hierarchical clustering, and UMAP.
#
# Inputs:
#   data/processed/03_tokens.rds
#   data/processed/02_tweets_clean.rds
#   data/processed/04_sentiment_tweets.rds
#
# Outputs:
#   data/processed/09D_clusters.csv
#   data/processed/09D_cluster_top_words.csv
#   data/processed/09D_cluster_representative_tweets.csv
#   outputs/figures/cluster_umap.png
#   outputs/figures/cluster_dendrogram.png
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(tidyr)
library(quanteda)
library(quanteda.textstats)
library(quanteda.textplots)
library(cluster)
library(ggplot2)
library(uwot)
library(readr)
library(here)
library(forcats)

# ---- 1. Load required datasets ----

tokens_path <- here("data", "processed", "03_tokens.rds")
clean_path  <- here("data", "processed", "02_tweets_clean.rds")
sent_path   <- here("data", "processed", "04_sentiment_tweets.rds")

if (!file.exists(tokens_path)) stop("03_tokens.rds missing.")
if (!file.exists(clean_path))  stop("02_tweets_clean.rds missing.")
if (!file.exists(sent_path))   stop("04_sentiment_tweets.rds missing.")

tokens <- readRDS(tokens_path)
clean  <- readRDS(clean_path)
sent   <- readRDS(sent_path)

message("Loaded tokens: ", nrow(tokens))
message("Loaded cleaned tweets: ", nrow(clean))
message("Loaded sentiment tweets: ", nrow(sent))

# ---- 2. Build quanteda corpus and DTM ----

message("Building quanteda corpus...")

corpus_q <- corpus(
  clean,
  text_field = "text_lower",
  docid_field = "status_id"
)

# ---- 2. Build TWO DFMs: raw-count + TF-IDF ----

message("Constructing DFM (raw + tfidf)...")

# build small aggregated corpus from tokens
corpus_for_dfm <- tokens %>%
  as_tibble() %>%
  group_by(status_id) %>%
  summarise(text = paste(word, collapse = " "), .groups = "drop") %>%
  corpus(text_field = "text", docid_field = "status_id")

# RAW count dfm for keyness
dfm_raw <- corpus_for_dfm %>%
  tokens(remove_punct = TRUE) %>%
  dfm() %>%
  dfm_trim(min_docfreq = 20)

# TF-IDF dfm for clustering
dfm_tfidf <- dfm_raw %>%
  dfm_tfidf()

message("DFM_raw dims:  ", ndoc(dfm_raw), " documents × ", nfeat(dfm_raw))
message("DFM_tfidf dims:", ndoc(dfm_tfidf), " documents × ", nfeat(dfm_tfidf))

# matrix for clustering
mat <- as.matrix(dfm_tfidf)

message("DFM dimensions: ", ndoc(dfm_all), " documents × ", nfeat(dfm_all), " features")

# Convert to matrix for clustering
mat <- as.matrix(dfm_all)

# ---- 3. K-means clustering (K = 8 default) ----

K <- 8
message("Running k-means with K = ", K, "...")

set.seed(123)
kmod <- kmeans(mat, centers = K, iter.max = 50, nstart = 10)

cluster_assign <- tibble(
  status_id = as.numeric(rownames(mat)),
  cluster = factor(kmod$cluster)
)

# ---- 4. Top words per cluster (corrected: one-vs-rest keyness) ----

message("Computing top keyness words per cluster...")

top_words_list <- list()

# ---- 4. Top words per cluster (robust version: auto-detect keyness column) ----

message("Computing top keyness words per cluster...")

top_words_list <- list()

for (cl in levels(cluster_assign$cluster)) {

  target_vec <- cluster_assign$cluster == cl

  kws_raw <- textstat_keyness(dfm_raw, target = target_vec) %>%
    as_tibble()

  # detect scoring column automatically
  score_col <- intersect(c("stat", "G2", "chi2", "log_likelihood", "likelihood"), names(kws_raw))

  if (length(score_col) == 0) stop("No keyness score column found.")

  score_col <- score_col[1]  # choose the first available

  kws <- kws_raw %>%
    slice_max(order_by = .data[[score_col]], n = 15) %>%
    mutate(cluster = cl)

  top_words_list[[cl]] <- kws
}

top_words <- bind_rows(top_words_list)
message("Top keyness words computed for all clusters.")

top_words <- bind_rows(top_words_list)

message("Top keyness words computed for all clusters.")

# ---- 5. Representative tweets per cluster ----

message("Selecting representative tweets...")

clean_small <- clean %>% select(status_id, text)

rep_tweets <- cluster_assign %>%
  left_join(clean_small, by = "status_id") %>%
  group_by(cluster) %>%
  slice_head(n = 10) %>%
  ungroup()

# ---- 6. Sentiment profiles per cluster ----

message("Computing cluster × sentiment summary...")

sent_cluster <- cluster_assign %>%
  left_join(sent %>% select(status_id, compound, afinn_sum), by = "status_id") %>%
  group_by(cluster) %>%
  summarise(
    n_tweets = n(),
    mean_compound = mean(compound, na.rm = TRUE),
    sd_compound   = sd(compound, na.rm = TRUE),
    mean_afinn    = mean(afinn_sum, na.rm = TRUE),
    sd_afinn      = sd(afinn_sum, na.rm = TRUE),
    .groups = "drop"
  )

# ---- 7. UMAP embedding for 2-D visualisation ----

message("Running UMAP projection...")

set.seed(101)
umap_emb <- umap(mat, n_neighbors = 15, min_dist = 0.1)

umap_df <- tibble(
  status_id = as.numeric(rownames(mat)),
  UMAP1 = umap_emb[,1],
  UMAP2 = umap_emb[,2]
) %>%
  left_join(cluster_assign, by = "status_id")

umap_fig <- ggplot(umap_df, aes(UMAP1, UMAP2, color = cluster)) +
  geom_point(alpha = 0.5, size = 1) +
  labs(
    title = "UMAP clustering of tweet themes",
    x = "UMAP 1", y = "UMAP 2"
  ) +
  theme_minimal()

umap_path <- here("outputs", "figures", "cluster_umap.png")
ggsave(umap_path, umap_fig, width = 9, height = 6, dpi = 300)

message("Saved UMAP figure to: ", umap_path)

# ---- 8. Hierarchical clustering dendrogram (sampled for feasibility) ----

message("Running hierarchical clustering on a 5k sample (feasible)...")

set.seed(2025)
sample_size <- 5000

sample_ids <- sample(rownames(mat), sample_size)

mat_sample <- mat[sample_ids, , drop = FALSE]

# PCA to reduce to 30 dimensions
pca_small <- prcomp(mat_sample, rank. = 30)

# distance matrix on 5k × 5k → ~200MB, feasible
dist_small <- dist(pca_small$x)

hclust_obj <- hclust(dist_small, method = "ward.D2")

# Save dendrogram
dend_path <- here("outputs", "figures", "cluster_dendrogram.png")
png(filename = dend_path, width = 900, height = 800)
plot(hclust_obj, labels = FALSE, main = "Hierarchical Dendrogram of Tweet Themes (5k sample)")
dev.off()

message("Saved dendrogram to: ", dend_path)

# ---- 9. Save results ----

clusters_out <- cluster_assign %>%
  left_join(clean %>% select(status_id, text), by = "status_id") %>%
  left_join(sent %>% select(status_id, compound, afinn_sum), by = "status_id")

cluster_path <- here("data", "processed", "09D_clusters.csv")
write_csv(clusters_out, cluster_path)

topw_path <- here("data", "processed", "09D_cluster_top_words.csv")
write_csv(top_words, topw_path)

rep_path <- here("data", "processed", "09D_cluster_representative_tweets.csv")
write_csv(rep_tweets, rep_path)

sentprof_path <- here("data", "processed", "09D_cluster_sentiment_profiles.csv")
write_csv(sent_cluster, sentprof_path)

# ---- 10. Final console report ----

message("\n------ 09D_theme_clustering Summary ------")
message("K-means clusters: ", K)
message("Cluster sizes:")
print(table(cluster_assign$cluster))
message("\nSentiment profiles:")
print(sent_cluster)

message("\nTop words per cluster saved to: ", topw_path)
message("Representative tweets saved to: ", rep_path)
message("Cluster assignments saved to: ", cluster_path)
message("Sentiment profiles saved to: ", sentprof_path)
message("Figures saved to: ", here("outputs", "figures"))
message("✅ 09D_theme_clustering.R completed successfully.")
# ==========================================================
# End of script
# ==========================================================

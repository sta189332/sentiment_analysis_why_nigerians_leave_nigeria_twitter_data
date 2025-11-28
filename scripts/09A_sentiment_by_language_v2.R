# ==========================================================
# Script: 09A_sentiment_by_language_v2.R
# Purpose:
#   Expanded sentiment × language analysis including UND
#   + statistical tests + enhanced plots.
# ==========================================================

if (!"here" %in% loadedNamespaces()) {
  source("scripts/00_setup_env.R")
}

library(dplyr)
library(ggplot2)
library(readr)
library(forcats)
library(tidyr)
library(rstatix)
library(here)

sent_path <- here("data", "processed", "04_sentiment_tweets.rds")
out_dir   <- here("outputs", "figures")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- 1. Load sentiment dataset ----
sent <- readRDS(sent_path)

message("Loaded sentiment dataset with ", nrow(sent), " tweets.")

# ---- 2. Clean language column ----
sent <- sent %>%
  mutate(
    lang_group = case_when(
      lang %in% c("en", "pcm") ~ lang,
      lang %in% c("und", "undetermined", "", NA) ~ "und",
      TRUE ~ lang
    )
  )

# reorder by frequency
sent$lang_group <- fct_reorder(sent$lang_group, sent$lang_group, .fun = length)

lang_dist <- sent %>%
  count(lang_group) %>%
  arrange(desc(n))

message("Language distribution:")
print(lang_dist)

write_csv(lang_dist, here("reports/lang_distribution.csv"))

# ---- 3. Sentiment summary by language ----
sent_by_lang <- sent %>%
  group_by(lang_group) %>%
  summarise(
    n_tweets       = n(),
    mean_afinn     = mean(afinn_sum, na.rm = TRUE),
    sd_afinn       = sd(afinn_sum, na.rm = TRUE),
    mean_compound  = mean(compound, na.rm = TRUE),
    sd_compound    = sd(compound, na.rm = TRUE),
    median_afinn   = median(afinn_sum, na.rm = TRUE),
    median_compound = median(compound, na.rm = TRUE),
    prop_afinn_neg  = mean(afinn_sum < 0, na.rm = TRUE),
    prop_comp_neg   = mean(compound < 0, na.rm = TRUE),
    .groups = "drop"
  )

summary_path <- here("reports/sentiment_by_language_summary_v2.csv")
write_csv(sent_by_lang, summary_path)

message("Extended language sentiment summary saved to: ", summary_path)

# ---- 4. NRC emotion summary ----
nrc_cols <- grep("^nrc_", names(sent), value = TRUE)

nrc_by_lang <- sent %>%
  group_by(lang_group) %>%
  summarise(across(all_of(nrc_cols), sum), .groups = "drop") %>%
  pivot_longer(-lang_group, names_to = "emotion", values_to = "count") %>%
  mutate(emotion = gsub("nrc_", "", emotion))

write_csv(nrc_by_lang, here("reports/nrc_by_language_summary_v2.csv"))

# ---- 5. Statistical Tests ----

## AFINN ANOVA & Kruskal–Wallis
anova_afinn  <- sent %>% anova_test(afinn_sum ~ lang_group)
kw_afinn     <- sent %>% kruskal_test(afinn_sum ~ lang_group)

## Compound ANOVA & Kruskal–Wallis
anova_comp   <- sent %>% anova_test(compound ~ lang_group)
kw_comp      <- sent %>% kruskal_test(compound ~ lang_group)

## Post-hoc Dunn if KW significant
posthoc_kw_comp  <- sent %>% dunn_test(compound ~ lang_group, p.adjust.method = "bonferroni")
posthoc_kw_afinn <- sent %>% dunn_test(afinn_sum ~ lang_group, p.adjust.method = "bonferroni")

write_csv(anova_afinn,  here("reports/anova_afinn_language.csv"))
write_csv(kw_afinn,     here("reports/kw_afinn_language.csv"))
write_csv(posthoc_kw_afinn, here("reports/posthoc_kw_afinn_language.csv"))

write_csv(anova_comp,   here("reports/anova_compound_language.csv"))
write_csv(kw_comp,      here("reports/kw_compound_language.csv"))
write_csv(posthoc_kw_comp, here("reports/posthoc_kw_compound_language.csv"))

message("Statistical tests saved (ANOVA, Kruskal-Wallis, post-hoc).")

# ---- 6. Plots ----

## Compound boxplot
p1 <- ggplot(sent, aes(lang_group, compound, fill = lang_group)) +
  geom_boxplot(outlier.alpha = 0.1) +
  labs(title = "Compound sentiment by language", x = "Language", y = "Compound sentiment") +
  theme_minimal()

ggsave(here("outputs/figures/lang_compound_boxplot_v2.png"),
       p1, width = 8, height = 5, dpi = 300)


## AFINN boxplot
p2 <- ggplot(sent, aes(lang_group, afinn_sum, fill = lang_group)) +
  geom_boxplot(outlier.alpha = 0.1) +
  labs(title = "AFINN sentiment by language", x = "Language", y = "AFINN score") +
  theme_minimal()

ggsave(here("outputs/figures/lang_afinn_boxplot_v2.png"),
       p2, width = 8, height = 5, dpi = 300)


## NRC emotion bar
p3 <- ggplot(nrc_by_lang, aes(emotion, count, fill = lang_group)) +
  geom_col(position = "dodge") +
  labs(title = "NRC emotion counts by language", x = "Emotion", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(here("outputs/figures/lang_nrc_emotions_bar_v2.png"),
       p3, width = 10, height = 6, dpi = 300)

# ---- 7. Final console summary ----
message("\n------ 09A_sentiment_by_language_v2 Summary ------")
print(sent_by_lang)
message("\nNRC emotion summary:")
print(head(nrc_by_lang))
message("\nANOVA + KW tests completed.")
message("Plots saved to: ", out_dir)
message("✅ 09A_sentiment_by_language_v2.R completed successfully.")

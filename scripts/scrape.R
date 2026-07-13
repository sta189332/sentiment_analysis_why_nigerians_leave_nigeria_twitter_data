library(rtweet)

## 1. Authenticate with API (token excluded for security)
bearer_token <- Sys.getenv("TWITTER_BEARER_TOKEN")
auth <- rtweet_app(bearer_token)

## 2. Define keyword set
keywords <- c(
"leave Nigeria", "left Nigeria", "why I left Nigeria",
"japa", "travel out", "relocation Nigeria",
"immigrate Nigeria", "migration Nigeria",
"Canada Nigeria", "UK Nigeria"
)

## 3. Query tweets across full temporal window
tweets_raw <- search_all_tweets(
query       = paste(keywords, collapse = " OR "),
start_tweets = "2018-01-01T00:00:00Z",
end_tweets   = "2025-10-31T23:59:59Z",
n = Inf,
token = auth
)

## 4. Save results
saveRDS(tweets_raw, "data/raw/why_nigerians_leave_2018_2025.rds")

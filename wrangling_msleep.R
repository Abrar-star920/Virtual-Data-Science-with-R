# =====================================================================
# Week 3: Data Wrangling and Preprocessing (Data Science With R)
# Dataset: msleep (ggplot2) - sleep patterns of 83 mammal species, 11 variables
# Run top to bottom in RStudio, or: Rscript wrangling_msleep.R
# Packages: install.packages(c("tidyverse", "skimr"))
# Outputs: data/raw, data/processed, figures/, sessionInfo.txt
# =====================================================================

# ---- 0. Setup ----------------------------------------------------
library(tidyverse)   # dplyr, tidyr, ggplot2, readr, stringr, forcats
library(skimr)       # compact data summaries

for (d in c("data/raw", "data/processed", "figures"))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

set.seed(2026)       # makes any random step reproducible
theme_set(theme_minimal(base_size = 12))

# ---- 1. Import and archive raw data ------------------------------
data("msleep", package = "ggplot2")
raw <- msleep                                  # keep an untouched copy
write_csv(raw, "data/raw/msleep_raw.csv")      # archive the raw data

glimpse(raw)
dim(raw)
head(raw, 8)
skim(raw)

# ---- 2.1 Missing values ------------------------------------------
missing_tbl <- raw |>
  summarise(across(everything(), ~ sum(is.na(.x)))) |>
  pivot_longer(everything(), names_to = "variable", values_to = "n_missing") |>
  mutate(pct_missing = round(100 * n_missing / nrow(raw), 1)) |>
  arrange(desc(n_missing))
missing_tbl

sum(complete.cases(raw))   # rows that would survive listwise deletion

fig1 <- ggplot(missing_tbl,
               aes(x = reorder(variable, pct_missing), y = pct_missing)) +
  geom_col(fill = "#2E75B6") +
  coord_flip() +
  labs(title = "Percentage of missing values by variable",
       x = NULL, y = "% missing")
ggsave("figures/w3_fig1_missing.png", fig1, width = 7, height = 4.5, dpi = 300)
fig1

# ---- 2.2 Types, levels, duplicates -------------------------------
sapply(raw, class)                 # data types
count(raw, vore)                   # feeding-type codes
count(raw, conservation)           # conservation-status codes
count(raw, order, sort = TRUE)     # taxonomic orders (many rare levels)
sum(duplicated(raw))               # exact duplicate rows
sum(duplicated(raw$name))          # duplicate species names

# ---- 2.3 Consistency checks --------------------------------------
# awake should equal 24 - sleep_total
raw |> filter(abs(awake - (24 - sleep_total)) > 0.1) |> nrow()

# REM sleep cannot exceed total sleep
raw |> filter(sleep_rem > sleep_total) |> nrow()

# measurements must be positive
raw |> filter(bodywt <= 0 | brainwt <= 0 | sleep_total <= 0) |> nrow()

# ---- 2.4 Outliers and skew ---------------------------------------
iqr_outlier <- function(x) {
  q <- quantile(x, c(0.25, 0.75), na.rm = TRUE)
  fence <- 1.5 * diff(q)
  x < q[1] - fence | x > q[2] + fence
}

raw |> summarise(across(where(is.numeric), ~ sum(iqr_outlier(.x), na.rm = TRUE)))

raw |> arrange(desc(bodywt)) |> select(name, bodywt, brainwt, sleep_total) |> head(5)

fig2 <- ggplot(raw, aes(x = bodywt)) +
  geom_histogram(bins = 40, fill = "#2E75B6", colour = "white") +
  labs(title = "Body weight before transformation",
       x = "Body weight (kg)", y = "Count")
ggsave("figures/w3_fig2_bodywt_raw.png", fig2, width = 7, height = 4.5, dpi = 300)
fig2

# ---- 3.1 Fix data types ------------------------------------------
clean <- raw |>
  mutate(
    name  = str_squish(name),
    vore  = factor(vore, levels = c("carni", "herbi", "insecti", "omni"),
                   labels = c("Carnivore", "Herbivore", "Insectivore", "Omnivore")),
    conservation = factor(conservation,
      levels = c("lc", "nt", "vu", "en", "cd", "domesticated"),
      labels = c("Least concern", "Near threatened", "Vulnerable", "Endangered",
                 "Conservation dependent", "Domesticated")),
    order = factor(order)
  ) |>
  distinct()                      # safeguard: no exact duplicates expected

# Safety check: recoding must not create new NAs
stopifnot(sum(is.na(clean$vore)) == sum(is.na(raw$vore)),
          sum(is.na(clean$conservation)) == sum(is.na(raw$conservation)))

# ---- 3.2 Handle missing values -----------------------------------
# (a) Categorical: make missingness an explicit level
clean <- clean |>
  mutate(vore = fct_na_value_to_level(vore, level = "Unknown"),
         conservation = fct_na_value_to_level(conservation, level = "Not recorded"))

# (b) Flag rows whose numeric values will be imputed
clean <- clean |>
  mutate(sleep_rem_imputed = is.na(sleep_rem),
         brainwt_imputed   = is.na(brainwt))

# (c) Brain weight: predict from body weight with a log-log regression
brain_model <- lm(log10(brainwt) ~ log10(bodywt), data = clean)
brain_pred  <- 10^predict(brain_model, newdata = clean)

# (d) REM sleep: median imputation, capped at total sleep
rem_median <- median(clean$sleep_rem, na.rm = TRUE)

clean <- clean |>
  mutate(
    brainwt   = if_else(is.na(brainwt), brain_pred, brainwt),
    sleep_rem = if_else(is.na(sleep_rem), pmin(rem_median, sleep_total), sleep_rem)
  ) |>
  select(-sleep_cycle, -awake)   # >50% missing / redundant (24 - sleep_total)

# ---- 4.1 Feature extraction --------------------------------------
clean <- clean |>
  mutate(
    log_bodywt  = log10(bodywt),
    log_brainwt = log10(brainwt),
    brain_pct   = 100 * brainwt / bodywt,     # brain mass as % of body mass
    rem_share   = sleep_rem / sleep_total,    # share of sleep spent in REM
    name_words  = str_count(name, "\\S+"),     # words in the common name
    sleep_pattern = factor(
      case_when(sleep_total < 8  ~ "Short",
                sleep_total < 14 ~ "Medium",
                TRUE             ~ "Long"),
      levels = c("Short", "Medium", "Long"), ordered = TRUE),
    order_grp = fct_lump_n(order, n = 5, other_level = "Other")
  )

count(clean, order_grp, sort = TRUE)
count(clean, sleep_pattern)

# ---- 4.2 Encoding ------------------------------------------------
# Ordinal encoding: keeps the Short < Medium < Long ordering
clean <- clean |> mutate(sleep_pattern_ord = as.integer(sleep_pattern))

# One-hot encoding of nominal variables (full dummy set, no reference level)
onehot <- model.matrix(
  ~ vore + order_grp - 1, data = clean,
  contrasts.arg = list(
    vore      = contrasts(clean$vore, contrasts = FALSE),
    order_grp = contrasts(clean$order_grp, contrasts = FALSE))
) |> as_tibble()

names(onehot) <- names(onehot) |> str_replace_all("[^A-Za-z0-9]+", "_")

dim(onehot)
names(onehot)

# ---- 4.3 Scaling -------------------------------------------------
minmax <- function(x) (x - min(x)) / (max(x) - min(x))

clean <- clean |>
  mutate(
    across(c(sleep_total, sleep_rem, log_bodywt, log_brainwt),
           ~ as.numeric(scale(.x)), .names = "{.col}_z"),
    across(c(rem_share, brain_pct), minmax, .names = "{.col}_mm")
  )

clean |> summarise(across(ends_with("_z"), list(mean = mean, sd = sd)))
clean |> summarise(across(ends_with("_mm"), list(min = min, max = max)))

# ---- 4.4 Leakage-safe scaling pattern ----------------------------
# Leakage-safe pattern for modelling: learn scaling from TRAIN only
idx   <- sample(seq_len(nrow(clean)), size = round(0.8 * nrow(clean)))
train <- clean[idx, ]
test  <- clean[-idx, ]

mu  <- mean(train$log_bodywt)
sdv <- sd(train$log_bodywt)
train$log_bodywt_z <- (train$log_bodywt - mu) / sdv
test$log_bodywt_z  <- (test$log_bodywt  - mu) / sdv   # reuse TRAIN statistics

# ---- 5. Validation and before/after ------------------------------
final <- bind_cols(clean, onehot)

stopifnot(
  nrow(final) == nrow(raw),
  sum(is.na(final[, c("vore", "conservation", "sleep_rem", "brainwt")])) == 0,
  all(final$bodywt > 0, final$brainwt > 0),
  all(final$sleep_rem <= final$sleep_total),
  all(is.finite(final$log_bodywt), is.finite(final$log_brainwt))
)

comparison <- tibble(
  stage         = c("raw", "final"),
  rows          = c(nrow(raw), nrow(final)),
  columns       = c(ncol(raw), ncol(final)),
  missing_cells = c(sum(is.na(raw)), sum(is.na(final)))
)
comparison

head(raw, 5)
final |> select(name, vore, conservation, sleep_total, sleep_rem, brainwt,
                log_bodywt, rem_share, sleep_pattern) |> head(5)

# ---- 6. Figures after cleaning -----------------------------------
fig3 <- ggplot(final, aes(x = log_bodywt)) +
  geom_histogram(bins = 30, fill = "#1F3864", colour = "white") +
  labs(title = "Body weight after log10 transformation",
       x = "log10(body weight in kg)", y = "Count")
ggsave("figures/w3_fig3_bodywt_log.png", fig3, width = 7, height = 4.5, dpi = 300)
fig3

fig4 <- ggplot(final, aes(x = log_bodywt, y = log_brainwt, colour = brainwt_imputed)) +
  geom_point(size = 2.2, alpha = 0.85) +
  scale_colour_manual(values = c(`FALSE` = "#2E75B6", `TRUE` = "#C00000"),
                      labels = c("Observed", "Imputed"), name = NULL) +
  labs(title = "Brain versus body weight (log10), observed and imputed values",
       x = "log10(body weight, kg)", y = "log10(brain weight, kg)")
ggsave("figures/w3_fig4_imputation.png", fig4, width = 7, height = 4.5, dpi = 300)
fig4

# ---- 7. Export and session info ----------------------------------
write_csv(final, "data/processed/msleep_clean.csv")
saveRDS(final, "data/processed/msleep_clean.rds")   # preserves factor types
writeLines(capture.output(sessionInfo()), "sessionInfo.txt")
sessionInfo()

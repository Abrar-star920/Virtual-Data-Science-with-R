# =====================================================================
# Week 2: Exploratory Data Analysis and Visualisation (Data Science With R)
# Dataset: diamonds (ggplot2) - about 54,000 diamonds, 10 variables
# Run top to bottom in RStudio. Figures are saved to ./figures/
# Packages: install.packages(c("tidyverse", "skimr", "scales"))
# =====================================================================

# ---- 0. Setup ----------------------------------------------------
library(tidyverse)   # dplyr, ggplot2, readr, tidyr, forcats
library(skimr)       # compact data summaries
library(scales)      # axis label formatting

dir.create("figures", showWarnings = FALSE)
theme_set(theme_minimal(base_size = 12))

# ---- 1. Import and first look ------------------------------------
data("diamonds", package = "ggplot2")

glimpse(diamonds)   # structure: types and first values
dim(diamonds)       # rows and columns
head(diamonds)
skim(diamonds)      # missingness, distribution summaries

# ---- 2. Data quality checks --------------------------------------
colSums(is.na(diamonds))                        # missing values
sum(duplicated(diamonds))                       # exact duplicate rows
diamonds |> filter(x == 0 | y == 0 | z == 0) |> nrow()   # impossible sizes
summary(diamonds[, c("x", "y", "z")])
diamonds |>
  filter(y > 20 | z > 10) |>                    # extreme dimensions
  select(carat, x, y, z, depth, price)

# ---- 3. Cleaning and preparation ---------------------------------
n_before <- nrow(diamonds)

diamonds_clean <- diamonds |>
  distinct() |>                       # drop exact duplicates
  filter(x > 0, y > 0, z > 0) |>      # drop impossible zero dimensions
  filter(y < 20, z < 10) |>           # drop extreme data-entry errors
  mutate(
    price_per_carat = price / carat,
    carat_band = base::cut(carat, breaks = c(0, 0.5, 1, 1.5, 2, Inf),
                           labels = c("<=0.5", "0.5-1", "1-1.5", "1.5-2", ">2")),
    log_price = log10(price)
  )

c(before = n_before, after = nrow(diamonds_clean),
  removed = n_before - nrow(diamonds_clean))

str(diamonds_clean$cut)   # already an ordered factor: Fair < ... < Ideal

# ---- 4. Summary statistics ---------------------------------------
num_vars <- c("carat", "depth", "table", "price", "x", "y", "z")

diamonds_clean |> select(all_of(num_vars)) |> summary()

diamonds_clean |>
  summarise(across(c(carat, price),
                   list(mean = mean, median = median, sd = sd)))

diamonds_clean |>
  group_by(cut) |>
  summarise(n = n(), median_carat = median(carat),
            median_price = median(price), mean_price = mean(price),
            .groups = "drop")

count(diamonds_clean, color)
count(diamonds_clean, clarity)

cor_mat <- diamonds_clean |> select(all_of(num_vars)) |> cor()
round(cor_mat, 2)

# Elasticity: % change in price for a 1% change in carat
coef(lm(log10(price) ~ log10(carat), data = diamonds_clean))

# ---- 5. Figure 1: bar chart of cut -------------------------------
p1 <- ggplot(diamonds_clean, aes(x = cut, fill = cut)) +
  geom_bar(show.legend = FALSE) +
  geom_text(stat = "count", aes(label = comma(after_stat(count))), vjust = -0.4) +
  scale_fill_brewer(palette = "Blues") +
  labs(title = "Number of diamonds by cut quality",
       x = "Cut (worst to best)", y = "Count")

ggsave("figures/fig1_cut_bar.png", p1, width = 7, height = 4.5, dpi = 300)
p1

# ---- 6. Figure 2: histogram of price -----------------------------
p2 <- ggplot(diamonds_clean, aes(x = price)) +
  geom_histogram(binwidth = 500, fill = "#2E75B6", colour = "white") +
  geom_vline(xintercept = median(diamonds_clean$price), colour = "#C00000",
             linetype = "dashed") +
  scale_x_continuous(labels = label_dollar()) +
  labs(title = "Distribution of diamond prices",
       subtitle = "Dashed line = median", x = "Price (USD)", y = "Count")

ggsave("figures/fig2_price_hist.png", p2, width = 7, height = 4.5, dpi = 300)
p2

# ---- 7. Figure 3: histogram of log price -------------------------
p3 <- ggplot(diamonds_clean, aes(x = log_price)) +
  geom_histogram(bins = 50, fill = "#1F3864", colour = "white") +
  labs(title = "Distribution of log10(price)",
       x = "log10(price in USD)", y = "Count")

ggsave("figures/fig3_logprice_hist.png", p3, width = 7, height = 4.5, dpi = 300)
p3

# ---- 8. Figure 4: box plot of price by cut -----------------------
p4 <- ggplot(diamonds_clean, aes(x = cut, y = price, fill = cut)) +
  geom_boxplot(show.legend = FALSE, outlier.alpha = 0.2) +
  scale_y_log10(labels = label_dollar()) +
  scale_fill_brewer(palette = "Blues") +
  labs(title = "Price by cut quality", x = "Cut", y = "Price (USD, log scale)")

ggsave("figures/fig4_price_by_cut.png", p4, width = 7, height = 4.5, dpi = 300)
p4

# ---- 9. Figure 5: box plot of carat by cut -----------------------
p5 <- ggplot(diamonds_clean, aes(x = cut, y = carat, fill = cut)) +
  geom_boxplot(show.legend = FALSE, outlier.alpha = 0.2) +
  scale_fill_brewer(palette = "Blues") +
  labs(title = "Carat weight by cut quality", x = "Cut", y = "Carat")

ggsave("figures/fig5_carat_by_cut.png", p5, width = 7, height = 4.5, dpi = 300)
p5

# ---- 10. Figure 6: scatter plot ----------------------------------
p6 <- ggplot(diamonds_clean, aes(x = carat, y = price)) +
  geom_point(alpha = 0.08, size = 0.7, colour = "#2E75B6") +
  geom_smooth(method = "lm", colour = "#C00000", se = FALSE) +
  scale_x_log10() +
  scale_y_log10(labels = label_dollar()) +
  labs(title = "Price versus carat (both axes log scale)",
       x = "Carat (log scale)", y = "Price (USD, log scale)")

ggsave("figures/fig6_carat_price_scatter.png", p6, width = 7, height = 4.5, dpi = 300)
p6

# ---- 11. Figure 7: correlation heat map --------------------------
cor_long <- as.data.frame(as.table(cor_mat)) |>
  rename(var1 = Var1, var2 = Var2, r = Freq)

p7 <- ggplot(cor_long, aes(x = var1, y = var2, fill = r)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = round(r, 2)), size = 3.2) +
  scale_fill_gradient2(low = "#B2182B", mid = "white", high = "#2166AC",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Correlation matrix of numeric variables",
       x = NULL, y = NULL, fill = "r")

ggsave("figures/fig7_correlation_heatmap.png", p7, width = 7, height = 5, dpi = 300)
p7

# ---- 12. Figure 8: price-per-carat heat map ----------------------
heat <- diamonds_clean |>
  group_by(color, clarity) |>
  summarise(med_ppc = median(price_per_carat), .groups = "drop")

p8 <- ggplot(heat, aes(x = color, y = clarity, fill = med_ppc)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = round(med_ppc)), size = 3) +
  scale_fill_viridis_c(option = "C", labels = label_dollar()) +
  labs(title = "Median price per carat by colour and clarity",
       subtitle = "Colour: D (best) to J (worst); clarity: I1 (worst) to IF (best)",
       x = "Colour grade", y = "Clarity grade", fill = "USD / carat")

ggsave("figures/fig8_ppc_heatmap.png", p8, width = 7, height = 5, dpi = 300)
p8

# ---- 13. Session info --------------------------------------------
sessionInfo()   # record package versions for reproducibility

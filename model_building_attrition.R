# =====================================================================
# Week 4: Model Building and Predictive Analysis (Data Science With R)
# Problem: predict employee attrition (IBM HR data, modeldata::attrition)
# Models: logistic regression (glm), decision tree (rpart), random forest (ranger)
# Packages: install.packages(c("dplyr", "ggplot2", "modeldata", "ranger", "rpart.plot"))
# Run top to bottom in RStudio. Figures -> figures/, results -> output/
# =====================================================================

# ---- 0. Setup ----------------------------------------------------
library(dplyr)        # data wrangling
library(ggplot2)      # plotting
library(modeldata)    # the attrition dataset
library(ranger)       # random forest
library(rpart)        # decision tree
library(rpart.plot)   # decision-tree plot

for (d in c("figures", "output")) dir.create(d, showWarnings = FALSE)
set.seed(2026)
theme_set(theme_minimal(base_size = 12))

# ---- 1. Data and target ------------------------------------------
data(attrition, package = "modeldata")
attrition <- as.data.frame(attrition)
# Levels c("No", "Yes"): glm() then models the probability of "Yes" (leaving)
attrition$Attrition <- factor(attrition$Attrition, levels = c("No", "Yes"))

dim(attrition)
glimpse(attrition)
sum(is.na(attrition))                          # missing values
table(attrition$Attrition)
round(100 * prop.table(table(attrition$Attrition)), 1)
max(prop.table(table(attrition$Attrition)))    # accuracy of always predicting "No"

p_balance <- ggplot(attrition, aes(x = Attrition, fill = Attrition)) +
  geom_bar(show.legend = FALSE) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.4) +
  scale_fill_manual(values = c(Yes = "#C00000", No = "#2E75B6")) +
  labs(title = "Class balance of the target variable",
       x = "Left the company?", y = "Employees")
ggsave("figures/w4_fig1_balance.png", p_balance, width = 6, height = 4, dpi = 300)
p_balance

# ---- 1b. Raw rates for reference ---------------------------------
# Raw attrition rates: used later to confirm the direction of model effects
attrition |> group_by(OverTime) |>
  summarise(n = n(), attrition_rate = round(mean(Attrition == "Yes"), 3))
attrition |> group_by(MaritalStatus) |>
  summarise(n = n(), attrition_rate = round(mean(Attrition == "Yes"), 3))

# ---- 2. Stratified split and cross-validation folds --------------
# Stratified 75/25 split: sample within each class so both sets keep ~16% leavers
set.seed(2026)
idx_yes <- which(attrition$Attrition == "Yes")
idx_no  <- which(attrition$Attrition == "No")
train_idx <- c(sample(idx_yes, round(0.75 * length(idx_yes))),
               sample(idx_no,  round(0.75 * length(idx_no))))
train <- attrition[train_idx, ]
test  <- attrition[-train_idx, ]

c(train = nrow(train), test = nrow(test))
round(100 * prop.table(table(train$Attrition)), 1)
round(100 * prop.table(table(test$Attrition)), 1)

# Stratified 10-fold cross-validation labels (for the TRAINING set only)
make_folds <- function(y, k = 10) {
  fold <- integer(length(y))
  for (cl in levels(y)) {
    ii <- which(y == cl)
    fold[ii] <- sample(rep(seq_len(k), length.out = length(ii)))
  }
  fold
}
set.seed(2026)
fold_id <- make_folds(train$Attrition, k = 10)
table(fold_id, train$Attrition)

# ---- 3. Preprocessing --------------------------------------------
# Ordered rating variables (satisfaction, involvement...) -> integer scores 1, 2, 3...
# Nothing is learned from the data here, so it is identical for train and test.
ordered_cols <- names(attrition)[sapply(attrition, is.ordered)]
ordered_cols

prep_data <- function(d) {
  d[ordered_cols] <- lapply(d[ordered_cols], as.integer)
  d
}
train <- prep_data(train)
test  <- prep_data(test)

sapply(train, class)

# ---- 4. Model functions and tuning grids -------------------------
# One function to FIT each model, one to PREDICT the probability of leaving
fit_lr   <- function(d, pr = NULL) glm(Attrition ~ ., data = d, family = binomial)
fit_tree <- function(d, pr) rpart(Attrition ~ ., data = d, method = "class",
  control = rpart.control(cp = pr$cp, maxdepth = pr$maxdepth, minsplit = pr$minsplit))
fit_rf   <- function(d, pr) ranger(Attrition ~ ., data = d, num.trees = 500,
  mtry = pr$mtry, min.node.size = pr$min.node.size,
  probability = TRUE, importance = "impurity", seed = 2026)

pred_lr   <- function(m, nd) suppressWarnings(predict(m, newdata = nd, type = "response"))
pred_tree <- function(m, nd) predict(m, newdata = nd, type = "prob")[, "Yes"]
pred_rf   <- function(m, nd) predict(m, data = nd)$predictions[, "Yes"]

# Hyperparameter grids (every combination is cross-validated)
tree_grid <- expand.grid(cp = c(0.001, 0.003, 0.01, 0.03),
                         maxdepth = c(3, 5, 8), minsplit = c(10, 30))
rf_grid   <- expand.grid(mtry = c(3, 6, 10, 15, 20),
                         min.node.size = c(1, 5, 10, 20))
nrow(tree_grid)   # 24 settings
nrow(rf_grid)     # 20 settings

# ---- 4b. Evaluation helper functions -----------------------------
# ROC AUC = probability a random leaver gets a higher score than a random stayer
auc_roc <- function(y, p) {
  r <- rank(p)                       # average ranks handle ties
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# PR AUC as average precision: mean precision at each true leaver in the ranking
auc_pr <- function(y, p) {
  o <- order(p, decreasing = TRUE); y <- y[o]
  prec <- cumsum(y) / seq_along(y)
  sum(prec[y == 1]) / sum(y)
}

# Accuracy, recall, specificity, precision, F1 and F2 at a probability threshold
class_metrics <- function(y, p, thr = 0.5) {
  pred <- as.integer(p >= thr)
  tp <- sum(pred == 1 & y == 1); fp <- sum(pred == 1 & y == 0)
  fn <- sum(pred == 0 & y == 1); tn <- sum(pred == 0 & y == 0)
  recall    <- tp / (tp + fn)
  precision <- if (tp + fp == 0) NA_real_ else tp / (tp + fp)
  fbeta <- function(b) (1 + b^2) * precision * recall / (b^2 * precision + recall)
  data.frame(threshold = thr, accuracy = (tp + tn) / length(y), recall = recall,
             specificity = tn / (tn + fp), precision = precision,
             f1 = fbeta(1), f2 = fbeta(2))
}

round_df <- function(d, k = 3) {
  d[sapply(d, is.numeric)] <- round(d[sapply(d, is.numeric)], k)
  d
}

# Cross-validation: returns out-of-fold predictions and the AUC of every fold
cv_eval <- function(fit_fun, pred_fun, d, fold_id, pr) {
  k <- max(fold_id)
  oof <- numeric(nrow(d)); fold_auc <- numeric(k)
  for (i in seq_len(k)) {
    tr <- d[fold_id != i, ]; va <- d[fold_id == i, ]
    p  <- as.numeric(pred_fun(fit_fun(tr, pr), va))
    oof[fold_id == i] <- p
    fold_auc[i] <- auc_roc(as.integer(va$Attrition == "Yes"), p)
  }
  list(oof = oof, fold_auc = fold_auc)
}

# Grid search: cross-validate every row of the grid
tune_model <- function(grid, fit_fun, pred_fun, d, fold_id) {
  res <- lapply(seq_len(nrow(grid)), function(i) {
    cv <- cv_eval(fit_fun, pred_fun, d, fold_id, as.list(grid[i, ]))
    data.frame(grid[i, , drop = FALSE], cv_auc = mean(cv$fold_auc),
               cv_se = sd(cv$fold_auc) / sqrt(length(cv$fold_auc)))
  })
  do.call(rbind, res)
}

# ---- 5. Training and tuning --------------------------------------
y_train <- as.integer(train$Attrition == "Yes")

# Logistic regression: nothing to tune, just cross-validate
cv_lr <- cv_eval(fit_lr, pred_lr, train, fold_id, NULL)

# Decision tree: grid search over 24 settings
tree_res <- tune_model(tree_grid, fit_tree, pred_tree, train, fold_id)

# Random forest: grid search over 20 settings (takes a few minutes)
rf_res <- tune_model(rf_grid, fit_rf, pred_rf, train, fold_id)

head(tree_res[order(-tree_res$cv_auc), ], 5)
head(rf_res[order(-rf_res$cv_auc), ], 5)

fig_tune <- ggplot(rf_res, aes(x = mtry, y = cv_auc, colour = factor(min.node.size))) +
  geom_line() + geom_point(size = 2) +
  labs(title = "Random forest tuning: cross-validated ROC AUC",
       x = "mtry (predictors tried per split)", y = "Mean cross-validated ROC AUC",
       colour = "min.node.size")
ggsave("figures/w4_fig2_rf_tuning.png", fig_tune, width = 7, height = 4.5, dpi = 300)
fig_tune

# ---- 6. Select best settings -------------------------------------
best_tree <- tree_res[which.max(tree_res$cv_auc), ]
best_rf   <- rf_res[which.max(rf_res$cv_auc), ]
best_tree
best_rf

# ---- 7. Cross-validated comparison -------------------------------
cv_tree <- cv_eval(fit_tree, pred_tree, train, fold_id, as.list(best_tree))
cv_rf   <- cv_eval(fit_rf,   pred_rf,   train, fold_id, as.list(best_rf))

cv_row <- function(name, cv) {
  cm <- class_metrics(y_train, cv$oof, 0.5)
  data.frame(model = name,
             roc_auc = mean(cv$fold_auc),
             se = sd(cv$fold_auc) / sqrt(length(cv$fold_auc)),
             pr_auc = auc_pr(y_train, cv$oof),
             accuracy = cm$accuracy, sensitivity = cm$recall,
             specificity = cm$specificity)
}
cv_table <- round_df(rbind(cv_row("Logistic regression", cv_lr),
                           cv_row("Decision tree", cv_tree),
                           cv_row("Random forest", cv_rf)))
cv_table

fig_cv <- ggplot(cv_table, aes(x = model, y = roc_auc, fill = model)) +
  geom_col(show.legend = FALSE) +
  geom_errorbar(aes(ymin = roc_auc - se, ymax = roc_auc + se), width = 0.2) +
  coord_cartesian(ylim = c(0.5, 1)) +
  labs(title = "Cross-validated ROC AUC (mean +/- 1 SE)", x = NULL, y = "ROC AUC")
ggsave("figures/w4_fig3_cv_auc.png", fig_cv, width = 7, height = 4.5, dpi = 300)
fig_cv

# ---- 8. Paired comparison across folds ---------------------------
# Paired comparison of fold-level AUCs (same 10 folds for every model)
t.test(cv_rf$fold_auc, cv_lr$fold_auc,   paired = TRUE)   # forest vs logistic
t.test(cv_rf$fold_auc, cv_tree$fold_auc, paired = TRUE)   # forest vs tree

# ---- 9. Final fit and test evaluation ----------------------------
# Refit each model on ALL training data, then score the test set exactly once
final_lr   <- fit_lr(train)
final_tree <- fit_tree(train, as.list(best_tree))
final_rf   <- fit_rf(train, as.list(best_rf))

y_test <- as.integer(test$Attrition == "Yes")
p_lr   <- as.numeric(pred_lr(final_lr, test))
p_tree <- as.numeric(pred_tree(final_tree, test))
p_rf   <- as.numeric(pred_rf(final_rf, test))

test_row <- function(name, p) {
  cm <- class_metrics(y_test, p, 0.5)
  data.frame(model = name, roc_auc = auc_roc(y_test, p), pr_auc = auc_pr(y_test, p),
             accuracy = cm$accuracy, sensitivity = cm$recall,
             specificity = cm$specificity, precision = cm$precision,
             f1 = cm$f1, brier = mean((p - y_test)^2))
}
test_table <- round_df(rbind(test_row("Logistic regression", p_lr),
                             test_row("Decision tree", p_tree),
                             test_row("Random forest", p_rf)))
test_table

test_preds <- data.frame(
  model = rep(c("Logistic regression", "Decision tree", "Random forest"),
              each = nrow(test)),
  truth = rep(y_test, 3),
  p     = c(p_lr, p_tree, p_rf))

# ---- 10. ROC curves ----------------------------------------------
model_cols <- c("Logistic regression" = "#2E75B6", "Decision tree" = "#7F7F7F",
                "Random forest" = "#C00000")

roc_points <- function(y, p) {
  thr <- sort(unique(c(p, 0, 1)), decreasing = TRUE)
  t(sapply(thr, function(t) {
    pred <- p >= t
    c(fpr = sum(pred & y == 0) / sum(y == 0), tpr = sum(pred & y == 1) / sum(y == 1))
  }))
}
roc_df <- do.call(rbind, lapply(split(test_preds, test_preds$model), function(d) {
  r <- roc_points(d$truth, d$p)
  data.frame(model = d$model[1], fpr = r[, "fpr"], tpr = r[, "tpr"])
}))

fig_roc <- ggplot(roc_df, aes(x = fpr, y = tpr, colour = model)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  coord_equal() + scale_colour_manual(values = model_cols) +
  labs(title = "ROC curves on the test set", x = "1 - specificity (false positive rate)",
       y = "Sensitivity (recall)", colour = NULL)
ggsave("figures/w4_fig4_roc.png", fig_roc, width = 7, height = 5.5, dpi = 300)
fig_roc

# ---- 11. Precision-recall curves ---------------------------------
pr_points <- function(y, p) {
  thr <- sort(unique(p), decreasing = TRUE)
  t(sapply(thr, function(t) {
    pred <- p >= t
    tp <- sum(pred & y == 1)
    c(recall = tp / sum(y == 1), precision = tp / sum(pred))
  }))
}
pr_df <- do.call(rbind, lapply(split(test_preds, test_preds$model), function(d) {
  r <- pr_points(d$truth, d$p)
  data.frame(model = d$model[1], recall = r[, "recall"], precision = r[, "precision"])
}))

fig_pr <- ggplot(pr_df, aes(x = recall, y = precision, colour = model)) +
  geom_path(linewidth = 1) +
  geom_hline(yintercept = mean(y_test), linetype = "dashed") +
  coord_cartesian(ylim = c(0, 1)) + scale_colour_manual(values = model_cols) +
  labs(title = "Precision-recall curves on the test set",
       subtitle = "Dashed line = precision of random guessing (the attrition rate)",
       x = "Recall", y = "Precision", colour = NULL)
ggsave("figures/w4_fig5_pr.png", fig_pr, width = 7, height = 5, dpi = 300)
fig_pr

# ---- 12. Calibration ---------------------------------------------
cal_df <- test_preds |>
  group_by(model) |>
  mutate(bin = ceiling(rank(p, ties.method = "first") / n() * 8)) |>
  group_by(model, bin) |>
  summarise(predicted = mean(p), observed = mean(truth), .groups = "drop")

fig_cal <- ggplot(cal_df, aes(x = predicted, y = observed, colour = model)) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  geom_line() + geom_point(size = 2) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  scale_colour_manual(values = model_cols) +
  labs(title = "Calibration of predicted probabilities (test set)",
       x = "Mean predicted probability", y = "Observed attrition rate", colour = NULL)
ggsave("figures/w4_fig6_calibration.png", fig_cal, width = 7, height = 5.5, dpi = 300)
fig_cal

# ---- 13. Threshold selection -------------------------------------
# Choose the threshold from OUT-OF-FOLD training predictions (never the test set)
thr_tbl <- do.call(rbind, lapply(seq(0.05, 0.60, by = 0.025),
                                 function(t) class_metrics(y_train, cv_rf$oof, t)))
best_thr <- thr_tbl$threshold[which.max(thr_tbl$f2)]   # maximise F2 (recall-weighted)
best_thr

thr_long <- rbind(
  data.frame(threshold = thr_tbl$threshold, metric = "recall",    value = thr_tbl$recall),
  data.frame(threshold = thr_tbl$threshold, metric = "precision", value = thr_tbl$precision),
  data.frame(threshold = thr_tbl$threshold, metric = "f2",        value = thr_tbl$f2))

fig_thr <- ggplot(thr_long, aes(x = threshold, y = value, colour = metric)) +
  geom_line(linewidth = 1) + geom_vline(xintercept = best_thr, linetype = "dashed") +
  labs(title = "Random forest: metrics versus decision threshold (CV predictions)",
       x = "Probability threshold for predicting 'Yes'", y = "Value", colour = NULL)
ggsave("figures/w4_fig7_threshold.png", fig_thr, width = 7, height = 4.5, dpi = 300)
fig_thr

# Apply the chosen threshold to the test set once
round_df(rbind(class_metrics(y_test, p_rf, 0.5), class_metrics(y_test, p_rf, best_thr)))

# ---- 14. Confusion matrix ----------------------------------------
pred_default <- factor(ifelse(p_rf >= 0.5,      "Yes", "No"), levels = c("No", "Yes"))
pred_tuned   <- factor(ifelse(p_rf >= best_thr, "Yes", "No"), levels = c("No", "Yes"))

table(Predicted = pred_default, Actual = test$Attrition)   # default threshold 0.5
table(Predicted = pred_tuned,   Actual = test$Attrition)   # tuned threshold

cm_df <- as.data.frame(table(Predicted = pred_tuned, Actual = test$Attrition))
cm_plot <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "#2E75B6", guide = "none") +
  labs(title = "Random forest confusion matrix (tuned threshold)")
ggsave("figures/w4_fig8_confusion.png", cm_plot, width = 6, height = 4.5, dpi = 300)
cm_plot

# ---- 15. Interpretation ------------------------------------------
# Random forest importance (ranger impurity importance)
rf_imp <- data.frame(variable = names(final_rf$variable.importance),
                     importance = as.numeric(final_rf$variable.importance))
rf_imp <- head(rf_imp[order(-rf_imp$importance), ], 15)
fig_imp_rf <- ggplot(rf_imp, aes(x = importance, y = reorder(variable, importance))) +
  geom_col(fill = "#C00000") +
  labs(title = "Random forest: top 15 predictors", x = "Importance (impurity)", y = NULL)
ggsave("figures/w4_fig9_imp_rf.png", fig_imp_rf, width = 7, height = 5, dpi = 300)
fig_imp_rf

# Logistic regression: coefficient table and importance as |z statistic|
lr_coef <- as.data.frame(summary(final_lr)$coefficients)
lr_coef$term <- rownames(lr_coef)
lr_coef <- lr_coef[lr_coef$term != "(Intercept)", ]
lr_coef$importance <- abs(lr_coef$`z value`)
lr_top <- head(lr_coef[order(-lr_coef$importance), ], 15)
round_df(head(lr_coef[order(lr_coef$`Pr(>|z|)`), c("term", "Estimate", "z value", "Pr(>|z|)")], 10), 4)

fig_imp_lr <- ggplot(lr_top, aes(x = importance, y = reorder(term, importance))) +
  geom_col(fill = "#2E75B6") +
  labs(title = "Logistic regression: top 15 predictors", x = "|z statistic|", y = NULL)
ggsave("figures/w4_fig10_imp_lr.png", fig_imp_lr, width = 7, height = 5, dpi = 300)
fig_imp_lr

# Decision tree: draw the fitted tree
png("figures/w4_fig11_tree.png", width = 2200, height = 1500, res = 250)
rpart.plot(final_tree, roundint = FALSE)
dev.off()
rpart.plot(final_tree, roundint = FALSE)

# ---- 16. Fairness check ------------------------------------------
fair_df <- data.frame(Gender = test$Gender, Age = test$Age, truth = y_test, p = p_rf)
fair_df$pred <- as.integer(fair_df$p >= best_thr)
fair_df$age_band <- cut(fair_df$Age, c(0, 30, 40, Inf),
                        labels = c("30 or under", "31-40", "over 40"))

group_summary <- function(d, g) {
  do.call(rbind, lapply(split(d, d[[g]]), function(s) {
    tp <- sum(s$pred == 1 & s$truth == 1)
    data.frame(group = as.character(s[[g]][1]), n = nrow(s),
               attrition_rate = mean(s$truth), flagged_rate = mean(s$pred),
               recall = tp / sum(s$truth == 1),
               precision = if (sum(s$pred) == 0) NA_real_ else tp / sum(s$pred),
               auc = auc_roc(s$truth, s$p))
  }))
}
round_df(rbind(group_summary(fair_df, "Gender"), group_summary(fair_df, "age_band")))

# ---- 17. Export --------------------------------------------------
saveRDS(final_rf, "output/rf_final_model.rds")
write.csv(test_table, "output/test_metrics.csv", row.names = FALSE)
write.csv(cv_table,   "output/cv_metrics.csv",   row.names = FALSE)
writeLines(capture.output(sessionInfo()), "sessionInfo.txt")
sessionInfo()

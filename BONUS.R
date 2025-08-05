# ---------------------------------
# BONUS – Advanced Integration in R
# ---------------------------------

#  This script contains reusable components for:
# - model construction
# - evaluation
# - data reshaping
# - robustness testing
# - and preparing for reporting in .Rmd

library(dplyr)
library(ggplot2)
library(tidyr)
library(car)
library(broom) 

# -------------------------------------
# STEP 1: Build Logistic Model Function
# -------------------------------------
build_logit_model <- function(data) {
  glm(target_score ~ runtime + short_movie + release_year,
      data = data,
      family = "binomial")
}

# -------------------------------
# STEP 2: Evaluate Model Function
# -------------------------------
evaluate_logit_model <- function(model, data, threshold = 0.4) {
  predicted_probs <- predict(model, type = "response")
  actual <- data$target_score
  predicted_class <- ifelse(predicted_probs >= threshold, 1, 0)
  
  confusion <- table(Predicted = predicted_class, Actual = actual)
  
  TP <- ifelse("1" %in% rownames(confusion) && "1" %in% colnames(confusion), confusion["1", "1"], 0)
  TN <- ifelse("0" %in% rownames(confusion) && "0" %in% colnames(confusion), confusion["0", "0"], 0)
  FP <- ifelse("1" %in% rownames(confusion) && "0" %in% colnames(confusion), confusion["1", "0"], 0)
  FN <- ifelse("0" %in% rownames(confusion) && "1" %in% colnames(confusion), confusion["0", "1"], 0)
  
  accuracy  <- (TP + TN) / sum(confusion)
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  recall    <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  f1_score  <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
  
  list(
    confusion_matrix = confusion,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score
  )
}

# -------------------------------------------
# STEP 3: Robustness Check by Threshold Sweep
# -------------------------------------------
thresholds <- seq(0.3, 0.6, by = 0.05)
robustness_results <- data.frame()

for (t in thresholds) {
  metrics <- evaluate_logit_model(logit_model, netflix_data, threshold = t)
  robustness_results <- rbind(robustness_results, data.frame(
    Threshold = t,
    Accuracy = metrics$accuracy,
    Precision = metrics$precision,
    Recall = metrics$recall,
    F1_Score = metrics$f1_score
  ))
}

# Plot performance across thresholds
ggplot(robustness_results, aes(x = Threshold)) +
  geom_line(aes(y = Accuracy, color = "Accuracy")) +
  geom_line(aes(y = Precision, color = "Precision")) +
  geom_line(aes(y = Recall, color = "Recall")) +
  geom_line(aes(y = F1_Score, color = "F1 Score")) +
  labs(title = "Model Performance vs Threshold",
       y = "Score", color = "Metric") +
  theme_minimal()

# -----------------------------------------------------
# STEP 4: Additional Analysis Example with pivot_longer
# -----------------------------------------------------
# Suppose we have more than one score column
# ratings_long <- ratings %>%
#   pivot_longer(cols = c(imdb_score, tmdb_score),
#                names_to = "source",
#                values_to = "score")

# -------------------------------------
# STEP 5: Data Filtering with semi_join
# -------------------------------------
# Link only movies that have at least one genre assigned
# valid_movies <- titles %>%
#   filter(type == "MOVIE") %>%
#   semi_join(genres, by = "id")


# 🎬 Netflix Movie Success Prediction – Logistic Regression Analysis

**Author**: Ana Bojescu  
**Technologies**: R, dplyr, ggplot2, logistic regression, tidyverse

This project analyzes Netflix movie data to model and predict the probability that a movie achieves high IMDb scores (above 7.5), focusing particularly on runtime and release year. A logistic regression model was built and evaluated using real Netflix data.

---

## 🎯 Business Question

**What is the probability that a movie released in the last 5 years receives an IMDb score above 7.5 if its runtime is under 90 minutes?**

This question supports strategic decision-making for content acquisition and production.

---

## 📊 Dataset Description

The dataset includes 5 files:
- `titles.csv` – metadata about movies
- `ratings.csv` – IMDb scores
- `genres.csv`, `countries.csv`, `certifications.csv` – optional auxiliary data

Filtered to:
- Only `type == "MOVIE"`
- Only `release_year >= 2020`
- Only rows with complete `runtime` and `imdb_score`

New variables created:
- `target_score`: 1 if IMDb > 7.5, otherwise 0
- `short_movie`: 1 if runtime < 90 min, otherwise 0

---

## 📈 Exploratory Analysis

Visualizations include:
- Distribution of IMDb scores (histogram)
- Boxplots comparing score vs. movie length
- Bar plot of success rates (IMDb > 7.5) by short/long movies
- Scatterplot of runtime vs. score

Key Findings:
- Most movies are rated between 6–7.5
- Short movies (<90 min) have a slightly higher success rate (8.28%) than long ones (5.34%)
- Runtime has some predictive power, though weak

---

## 📦 Modeling Approach

Model used: **Binary Logistic Regression**
```r
glm(target_score ~ runtime + short_movie + release_year, family = "binomial")
```

Evaluation:
- All predictors were statistically significant
- McFadden pseudo R² = 0.0665
- MAE = 0.2107, RMSE = 0.3251
- Accuracy = 86.8%
- Precision = 43.4%, Recall = 7.5%, F1 Score = 12.7%

Conclusion:
- The model is precise but conservative (low recall, many TN)
- Runtime and release year positively impact success

---

## 🔍 Residual Analysis

Residuals plotted against runtime show no major pattern or bias, indicating reasonable model fit. However, residuals suggest prediction skew for borderline scores.

---

## 🚀 Model Testing on Hypothetical Data

A simulated dataset was created with fictional movies to test the model. Results showed that older movies (e.g., from 1980) had the highest predicted success probabilities.

### Sample Predictions:
- "Short Classic" (1980, 80min): 60% success chance
- "Short Modern" (2022, 85min): <20% success chance

---

## 💼 Executive Recommendation

**Title:** Prioritize Long, Recent Movies for Higher Success Rates

- Allocate production budgets to long-format movies (>100 min)
- Deprioritize short movies or outdated styles
- Use the model as a conservative risk filter – it avoids false positives

The model offers solid support for content development decisions based on historical performance indicators.

---

## 🧰 Tools & Libraries Used
- R Language
- dplyr, tidyverse
- ggplot2
- car (for VIF check)
- `glm()` base function for logistic modeling

---

## 🚀 How to Run This Project Locally

1. Clone the repository:
```bash
git clone 
cd netflix-logistic-analysis

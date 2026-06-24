# Recommender System - MovieLens
**Recommender Systems & Collaborative Filtering | Exploratory Data Analysis & Model Tuning**

### Project Overview
This project develops and evaluates collaborative filtering recommender systems using the **MovieLens dataset** (27M+ ratings). The objective was to build, tune, and compare multiple recommendation algorithms—specifically **KNNBaseline** and **SVD (Singular Value Decomposition)**—to predict user movie preferences accurately.

### Tech Stack & Methodology
- **Language:** Python
- **Core Library:** `Surprise` (Scikit-Surprise)
- **Algorithms:** 
    - **KNNBaseline:** Neighborhood-based approach using Pearson baseline similarity.
    - **SVD:** Matrix factorization identifying latent factors.
- **Validation:** 5-fold Cross-Validation with stratified sampling to ensure robust generalization.

### Pipeline Workflow
- **Data Preprocessing (`Part 1 & 2`):** Performed extensive EDA on 27M+ ratings. Implemented stratified sampling and top-K subsetting to optimize memory usage while preserving original data rating distributions (99% correlation retained).
- **Baseline Evaluation (`Part 3`):** Established `NormalPredictor` as the random baseline to benchmark algorithmic improvements.
- **Hyperparameter Tuning (`Part 4 & 5`):** Utilized `GridSearchCV` and `RandomizedSearchCV` to optimize critical parameters:
    - **KNNBaseline:** Tuned k, min_k, and similarity metrics (Pearson baseline/Cosine).
    - **SVD:** Optimized latent factors (n_factors), epochs, learning rates, and regularization strength to prevent overfitting.

### Key Optimization Insights
- **Scalability Handling:** Addressed memory constraints by switching from user-based to item-based collaborative filtering, allowing the `KNNBaseline` model to run efficiently on large-scale sparse matrices.
- **Model Performance:** 
    - **SVD** demonstrated superior scalability and achieved the lowest final RMSE of **0.8032** after tuning.
    - **KNNBaseline** provided excellent performance (**0.8143** RMSE) with higher interpretability regarding neighborhood similarities.
- **Robustness:** Verified generalization capability by re-running cross-validation with different random seeds (42 vs 101), showing minimal performance variance and confirming model robustness.

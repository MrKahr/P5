# Beginner Topics
 - Glossary of terms [https://scikit-learn.org/stable/glossary.html]
 - Explanation of Chi-square and Fischer's Exact [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5426219/]
 - Chi-square algo [https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html]


# Advanced Topics and some litterature overview
>Paper about **Conditional Predictive Impact**: a novel way to calculate Conditional Independence in supervised ML models [https://link.springer.com/article/10.1007/s10994-021-06030-6]
* 2.3 The Knock-off framework: Calculate False Discovery Rate (FDR) using:
    1. Markov Blanket. FDR is given as the expected proportion of false positives to true positives (p. 2111-2112).
    2. Adaptive Threshold test (ATT). Note: It has several issues, e.g., it requires a large amount of variables to reliably detect true positives (p. 2112).
    3. Conditional Randomization Test (CRT) - slightly better than ATT but computationally expensive (p. 2112).
    4. Conditional Predictive Impact (this paper).

* Calculate information gain based on actual variable and knock-off variable compared using a loss function (p. 2112-2113).
* Calculate the risk of model. I.e., how well it generalizes beyond its training data (p. 2113).


# High-level step-by-step guide
1. Find conditional independent variables using either:
    - Paper about CPI [https://link.springer.com/article/10.1007/s10994-021-06030-6]

4. Dimensionality reduction (removing unpredictive features) using either:
    - Recursive feature elimination with cross-validation [https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html]

5. Train model using:
    - Cross-validation

6. Calculate feature importance for the selected model using either:
    - Permutation importance [https://scikit-learn.org/stable/modules/permutation_importance.html]
    - Model-specific
        - Random Forest importance [https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html]


# Code examples
- Using Chi-squared [https://medium.com/machine-learning-t%C3%BCrkiye/decoding-patterns-in-categorical-data-independence-testing-101-c47934653f3e]
- Walkthrough of a simple ML example from data exploration to finished model [https://towardsdatascience.com/machine-learning-on-categorical-variables-3b76ffe4a7cb]
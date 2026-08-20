Yihang Feng

To what extent do the predictive gains of complex machine-learning models over logistic regression justify reduced interpretability in consumer credit scoring?

This essay argues that the predictive gains of complex machine-learning models only justify lower interpretability in some situations. They should be used when they perform clearly and consistently better than logistic regression in out-of-sample tests, and when the gain is important for real lending decisions. The model must also be possible to explain and check for fairness and model risk. If an interpretable model gives similar results, logistic regression or another interpretable model should be preferred.

section 1   How large and useful are the prediction gains? Decide how much improvement is enough and discuss the strongest case for complex models.

Firstly, the prediction gain should be large, stable and useful in real lending decisions before it can justify a less interpretable model.

Comparing 41 classifiers across eight credit-scoring datasets, Lessmann et al. (2015) found that several complex models outperformed logistic regression, with average simulated cost improvements of 3.4% for neural networks, 5.7% for random forests and 4.8% for HCES-Bag. However, model rankings changed when the evaluation metric or misclassification costs changed, suggesting that average gains alone do not justify sacrificing interpretability.

Khandani et al. (2010) estimate that using machine-learning predictions to reduce credit limits for high-risk accounts could save 6–25% of total losses under the authors' simulated assumptions. However, this estimate depends on richer transaction data and cost assumptions set by the authors, suggesting that not all of the estimated benefit can be attributed to model complexity.

Although even a modest predictive improvement may have substantial economic value for a large lender, this does not by itself justify using a less interpretable model. Therefore, reduced interpretability is justified only when the predictive gain remains stable in out-of-sample tests, across evaluation measures and under realistic cost assumptions.

Complex models are most useful when the dataset is very large and the relationships between variables are strongly nonlinear, but these situations may not represent all consumer credit scoring.

Using data on more than 120 million US mortgages, Sadhwani et al. (2021) found that a portfolio of 20,000 loans selected by a five-layer neural network had 46% lower one-year losses than a portfolio selected by the linear model under the authors' loss assumptions. This provides a strong case for complex models when datasets are extremely large and relationships are strongly nonlinear, although the result cannot be directly generalised to ordinary consumer credit scoring.

Section 2   Can the loss of interpretability be avoided or controlled? Compare complex models with interpretable alternatives and discuss post-hoc explanation.

However, lower interpretability may not always be necessary because some interpretable or hybrid models can also learn nonlinear relationships.

When a black-box model is still more accurate, post-hoc explanation can make it easier to check, but it cannot fully remove the uncertainty caused by the black box.

Section 3 Who benefits from the better predictions? Discuss fairness and the effects on different groups of borrowers.

Finally, a model should not be selected only by its overall accuracy because the prediction gains may not be shared equally between different groups.

Conclusion 

repeat the same idea with introduction

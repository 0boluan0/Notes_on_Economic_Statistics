---
title: LN905 Detailed Essay Plan
course: LN905
status: ready-to-submit
essay_question: To what extent do the predictive gains of complex machine-learning models over logistic regression justify reduced interpretability in consumer credit scoring?
---

# LN905 Detailed Essay Plan

## Essay question

To what extent do the predictive gains of complex machine-learning models over logistic regression justify reduced interpretability in consumer credit scoring?

## Working thesis

This essay argues that the predictive gains of complex machine-learning models only justify lower interpretability in some situations. They should be used when they perform clearly and consistently better than logistic regression in out-of-sample tests, and when the gain is important for real lending decisions. The model must also be possible to explain and check for fairness and model risk. If an interpretable model gives similar results, logistic regression or another interpretable model should be preferred.

## Structure and line of argument

**Structure:** Thematic. This structure is suitable because the answer is “it depends”. Each section checks one condition of the thesis.

1. **Section One — How large and useful are the prediction gains?** Decide how much improvement is enough and discuss the strongest case for complex models.
2. **Section Two — Can the loss of interpretability be avoided or controlled?** Compare complex models with interpretable alternatives and discuss post-hoc explanation.
3. **Section Three — Who benefits from the better predictions?** Discuss fairness and the effects on different groups of borrowers.

The argument moves through four tests:

**clear and useful gain → no interpretable model gives a similar result → the complex model can be checked → the effects on different groups are acceptable**

If a model does not meet one of these conditions, logistic regression or an interpretable alternative should be used. A complex model is justified only when it meets all four conditions.

## Introduction

**Function:** Introduce the problem, explain the main terms and give the answer.

- Explain that logistic regression is widely used because it is easier to understand and check. Complex models may be more accurate because they can find nonlinear relationships and interactions.
- Define **prediction gains** as improvements which remain in out-of-sample tests and lead to better lending decisions, not only a higher score on one measure.
- Define **interpretability** as the ability of banks, regulators and borrowers to understand and check how the model makes a decision. A post-hoc explanation is different because it explains a black box after the model has been trained.
- Give the thesis and briefly introduce the three sections.

## Section One — How large and useful are the prediction gains?

**Section function:** Explain what kind of improvement is strong enough to support a complex model, and then examine the strongest evidence for using one.

### Paragraph 1 — How much improvement is enough?

**Topic sentence:** Firstly, the prediction gain should be large, stable and useful in real lending decisions before it can justify a less interpretable model.

- **Evidence:** Lessmann et al. (2015) compare 41 models on eight datasets. Complex models often perform better than logistic regression, but the ranking changes when different measures and error costs are used. Their estimated cost improvements are about 3.4% to 5.7%. Khandani et al. (2010) estimate that machine learning may save 6–25% of total losses.
- **Evaluation/counterargument:** A small improvement may still be important for a large bank. However, Khandani et al. use richer transaction data and their savings depend on several assumptions. The improvement may come from better data, not only from a more complex model.
- **Paragraph result:** The gain should remain in out-of-sample tests, different measures and realistic cost settings.

### Paragraph 2 — When complex models are most useful

**Topic sentence:** Complex models are most useful when the dataset is very large and the relationships between variables are strongly nonlinear, but these situations may not represent all consumer credit scoring.

- **Evidence:** Sadhwani et al. (2021) use more than 120 million US mortgages. Their deep-learning model finds strong nonlinear relationships and reports 46% lower one-year portfolio losses than a linear model under the authors' assumptions. Bracke et al. (2019) find a smaller gain in UK mortgage default prediction: ROC-AUC increases from about 0.78 to 0.81.
- **Evaluation/counterargument:** These results may suggest that logistic regression is no longer useful. However, both studies focus on mortgages, Bracke et al. use a random test split, and Sadhwani et al. do not measure the costs of lower interpretability or fairness problems.
- **Paragraph result:** These studies show when complexity can be useful, but they do not support replacing logistic regression in every credit-scoring situation.

## Section Two — Can the loss of interpretability be avoided or controlled?

**Section function:** Show that the choice is not always between a simple but weak model and an accurate black box. Then discuss whether explanation methods are enough when a black box is still better.

### Paragraph 3 — Interpretable alternatives

**Topic sentence:** However, lower interpretability may not always be necessary because some interpretable or hybrid models can also learn nonlinear relationships.

- **Evidence:** Rudin (2019) argues that accuracy and interpretability do not always have a trade-off. In Bücker et al.'s (2022) case study, spline logistic regression has the best test result. Dumitrescu et al. (2022) combine short decision-tree rules with logistic regression; this model beats standard logistic regression on four datasets and is often close to random forests.
- **Evaluation/counterargument:** Hybrid models are not always easy for borrowers or regulators to understand, and random forest is still better on one dataset. However, they show that a black box should not only be compared with basic logistic regression.
- **Paragraph result:** A complex black box is justified only by the extra gain that remains after comparison with strong scorecards, spline models and interpretable hybrid models.

### Paragraph 4 — What post-hoc explanation can and cannot solve

**Topic sentence:** When a black-box model is still more accurate, post-hoc explanation can make it easier to check, but it cannot fully remove the uncertainty caused by the black box.

- **Evidence:** Bracke et al. (2019) show that an explanation method can identify the main factors behind general and individual predictions. However, the importance of the factors changes between loans and in stress tests. Bücker et al. (2022) therefore argue that explanation should be part of a wider audit of the model. Rudin (2019) also warns that an explanation may look reasonable but still not show the real logic of the black box.
- **Evaluation/counterargument:** Explanation tools can help banks check a model, especially when they are combined with validation and stress testing. However, explaining a model after it makes a prediction is not the same as directly understanding the model.
- **Paragraph result:** Post-hoc explanation reduces the problem but does not solve it completely, so the model still needs regular checking.

## Section Three — Who benefits from the better predictions?

**Section function:** Check whether higher overall accuracy also gives acceptable results for different groups of borrowers.

### Paragraph 5 — Fairness and distributional consequences

**Topic sentence:** Finally, a model should not be selected only by its overall accuracy because the prediction gains may not be shared equally between different groups.

- **Evidence:** Using data from 800,000 UK borrowers, Bono et al. (2021) find that ensemble models are more accurate than penalised logistic regression, but they neither clearly increase nor remove the existing fairness differences. Fuster et al. (2022) find that random forests improve overall prediction on nearly ten million US mortgages, but Black and Hispanic borrowers benefit less and interest-rate differences may increase.
- **Evaluation/counterargument:** Logistic regression can also be unfair, so a simple model does not guarantee fairness. However, a less interpretable model makes it harder to understand and challenge unequal effects. Also, Fuster et al.'s market results depend on assumptions about rejected applications and borrower behaviour.
- **Paragraph result:** Banks should test errors and lending outcomes for different groups instead of judging fairness from the model type or overall accuracy.

## Conclusion

**Function:** Give a direct answer and bring the three sections together.

- Restate that prediction gains justify lower interpretability only in some situations.
- Give the final decision process: start with a strong logistic regression model; compare it with interpretable nonlinear and hybrid models; use a black box only when its extra gain is clear and stable and it passes explanation, stress-testing and fairness checks.
- Conclude that complex models can be useful, but they should not automatically replace logistic regression in consumer credit scoring.

## Provisional references

Bono, T., Croxson, K., & Giles, A. (2021). Algorithmic fairness in credit scoring. *Oxford Review of Economic Policy, 37*(3), 585–617. https://doi.org/10.1093/oxrep/grab020

Bracke, P., Datta, A., Jung, C., & Sen, S. (2019). *Machine learning explainability in finance: An application to default risk analysis* (Staff Working Paper No. 816). Bank of England.

Bücker, M., Szepannek, G., Gosiewska, A., & Biecek, P. (2022). Transparency, auditability, and explainability of machine learning models in credit scoring. *Journal of the Operational Research Society, 73*(1), 70–90. https://doi.org/10.1080/01605682.2021.1922098

Dumitrescu, E., Hué, S., Hurlin, C., & Tokpavi, S. (2022). Machine learning for credit scoring: Improving logistic regression with non-linear decision-tree effects. *European Journal of Operational Research, 297*(3), 1178–1192. https://doi.org/10.1016/j.ejor.2021.06.053

Fuster, A., Goldsmith-Pinkham, P., Ramadorai, T., & Walther, A. (2022). Predictably unequal? The effects of machine learning on credit markets. *The Journal of Finance, 77*(1), 5–47. https://doi.org/10.1111/jofi.13090

Khandani, A. E., Kim, A. J., & Lo, A. W. (2010). Consumer credit-risk models via machine-learning algorithms. *Journal of Banking & Finance, 34*(11), 2767–2787. https://doi.org/10.1016/j.jbankfin.2010.06.001

Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research, 247*(1), 124–136. https://doi.org/10.1016/j.ejor.2015.05.030

Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence, 1*, 206–215. https://doi.org/10.1038/s42256-019-0048-x

Sadhwani, A., Giesecke, K., & Sirignano, J. (2021). Deep learning for mortgage risk. *Journal of Financial Econometrics, 19*(2), 313–368. https://doi.org/10.1093/jjfinec/nbaa025

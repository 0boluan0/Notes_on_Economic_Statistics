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

## Fast drafting blueprint｜完整参考蓝图

> [!warning] 使用边界
> 这是逐段改写用的论证蓝图，不是可直接提交的英文成稿。数字、来源关系和限制已经核对；正式正文仍需用自己的表达重写，并保留对每段判断的控制。

### Introduction｜约 200 词

按下面顺序写，不从文献数字开始：

1. **背景**：consumer credit scoring 是高风险决策；logistic regression 容易理解和审计，复杂机器学习可以捕捉 nonlinear relationships 与 interactions。
2. **问题张力**：更高预测表现可能减少违约损失，但更低可解释性会增加审计、监管、模型风险和借款人挑战决定的困难。
3. **定义 predictive gains**：不是单个指标偶尔上升，而是在真正样本外测试中保持稳定，并改善实际 lending decisions 的增益。
4. **定义 interpretability**：银行、监管者和借款人能理解并检查模型怎样得出决定；post-hoc explanation 是对黑箱结果的事后近似，不等于模型本身透明。
5. **直接回答程度**：复杂模型的收益只在部分情境足以抵偿可解释性下降。
6. **路线**：依次检查增益是否实质稳定、是否存在表现相近的可解释替代方案、黑箱能否被可靠审计，以及不同借款群体是否获得可接受的结果。

可调用句框：

- `In consumer credit scoring, the choice between ... is important because ...`
- `In this essay, predictive gains refer to ... rather than ...`
- `Interpretability refers to ..., whereas post-hoc explanation ...`
- `This essay argues that reduced interpretability is justified only when ...`

### P1｜收益必须实质、稳定并能改变真实决策｜约 320 词

这段已有学习者正文，改稿时只检查下面的完整链：

1. **Writer claim**：预测提升必须 large、stable、useful，才可能补偿透明度损失。
2. **Lessmann evidence**：41 个分类器、8 个 credit-scoring datasets；神经网络、随机森林和 HCES-Bag 相对 logistic regression 的平均模拟成本改善约 3.4%、5.7% 和 4.8%。
3. **Lessmann evaluation**：模型排名随评价指标和误分类成本变化；研究是 benchmark 与简化成本模拟，没有计入治理、解释、维护和监管成本。
4. **Khandani evidence**：利用更丰富的交易、余额和征信数据进行滚动样本外预测；在作者假设下，削减高风险账户额度可能节约总损失的 6%–25%。
5. **Khandani limitation**：单一银行、金融危机时期、成本收益假设；收益来自及时数据、特征构造和模型的组合，不能全部归因于算法复杂度。
6. **段尾判断**：小幅增益对大型机构也可能有价值，但 `average gains alone do not justify sacrificing interpretability`；增益必须跨样本外测试、指标和现实成本设定保持稳定。

### P2｜复杂度在超大规模和强非线性环境中理由最强｜约 320 词

1. **Writer claim**：复杂模型最有价值的情况，是数据极大且线性模型会遗漏重要阈值、交互和非线性关系；这不是所有消费者信用评分的默认情况。
2. **Sadhwani evidence**：超过 1.2 亿笔美国 mortgages、约 35 亿条 loan-month observations，并使用真正的时间切分；五层神经网络能捕捉 prepayment、FICO、LTV、利率和宏观变量之间的非线性与交互。
3. **Sadhwani decision value**：在作者损失假设下，神经网络选择的 20,000 笔贷款组合一年损失比线性模型组合低 46%。
4. **Sadhwani limitation**：这是 mortgage/MBS 风险和模拟投资组合，不是普通贷款申请 scorecard；没有直接计入公平性、解释和监管部署成本。
5. **Bracke contrast**：约 583 万笔英国 mortgages；gradient tree boosting 相对 logistic regression 的 ROC-AUC 约从 0.78 增至 0.81，precision-recall AUC 约从 0.16 增至 0.19，提升明显温和。
6. **Bracke limitation**：训练与测试随机切分，不是真正 out-of-time；违约率约 2.5%，precision 仍有限，范围仍是 mortgages。
7. **跨来源综合**：Sadhwani 是复杂度可能产生实质价值的边界强案例；Bracke 表明即使数据规模很大，增益也可能较小且受测试方法限制。
8. **段尾判断**：复杂度只有在具体数据环境持续产生 material and stable gains 时才更可能合理；两项研究都不能支持所有信用评分默认替换 logistic regression。

可调用句框：

- `In contrast, Bracke et al. (2019) report ...`
- `Taken together, the two studies suggest that ...`
- `The scale of the data alone does not justify ... because ...`

### P3｜接受黑箱前，先比较强可解释替代方案｜约 320 词

1. **Writer claim**：题目不应被处理成“基础 logistic regression 或高准确率黑箱”的二选一。
2. **Rudin principle**：高风险决策中不存在普遍、必然的 accuracy–interpretability trade-off；采用者应先证明没有表现相近的 intrinsically interpretable model。
3. **Rudin limitation**：这是跨领域理论与评论文章，不是统一的消费者信用评分 benchmark，因此适合提供举证原则，不适合单独证明具体性能差距。
4. **Bücker evidence**：在 10,459 个 HELOC 样本上，比较 scorecard、逻辑回归、spline logistic regression 与多种复杂模型；最佳测试结果来自 spline logistic regression。
5. **Bücker limitation**：单个公共数据集、随机切分，没有真正 out-of-time 测试；但它证明精心构造的可解释模型可能接近或超过黑箱。
6. **Dumitrescu evidence**：PLTR 用浅层树生成阈值/交互规则，再由 adaptive-lasso logistic regression 选择；四个数据集都优于标准 logistic regression。Kaggle 数据上 PLTR AUC 为 0.8568，略高于 random forest 的 0.8529，并明显高于线性 logistic regression 的 0.6983。
7. **Dumitrescu counterpoint**：Housing 数据上 random forest 仍明显更好；PLTR 的可解释性主要由规则数量和长度推断，未由真实借款人或监管者测试。
8. **段尾判断**：黑箱只能用它相对 strong scorecard、spline model 和 interpretable hybrid 的额外增益来辩护，而不能只通过击败最基础的 logistic regression 来辩护。

可调用句框：

- `This comparison weakens the assumption that ...`
- `Rather than comparing a black box only with ..., lenders should ...`
- `A loss of interpretability is therefore justified only by the additional gain that remains after ...`

### P4｜事后解释有用，但不能把黑箱变成透明模型｜约 320 词

1. **Writer claim**：如果黑箱仍明显更准，post-hoc explanation 可以减轻治理问题，但不能完全消除黑箱的不确定性。
2. **Bracke positive evidence**：QII/Shapley 方法能够显示总体与个体预测的重要变量；current LTV 和 current interest rate 等结果符合 mortgage 风险知识，局部解释还能展示不同贷款的异质性。
3. **Bracke boundary**：压力情景中变量影响排序明显变化；普通测试集中的合理解释不能证明模型在新状态仍可靠。解释描述预测关系，不建立因果。
4. **Bücker governance evidence**：TAX4CS 要求按 stakeholder、model lifetime、explanation need 和 XAI method 组织完整审计；一次 SHAP 图不等于治理。
5. **Bücker boundary**：该框架说明复杂模型可以被系统检查，但没有证明所有解释都忠实，也没有直接测量真实银行的维护成本或监管接受度。
6. **Rudin counterargument**：事后解释是对黑箱的近似，只要它更简单，就可能遗漏或歪曲原模型在部分输入区域的逻辑；看似合理不等于 faithful。
7. **让步**：解释工具与 validation、stress testing 和领域知识结合时，确实能支持异常诊断和 stakeholder communication。
8. **段尾判断**：post-hoc explanation reduces rather than removes the interpretability cost；黑箱仍需持续验证、压力测试、稳定性检查和明确责任。

可调用句框：

- `Post-hoc tools can reveal ..., but they cannot establish that ...`
- `A plausible explanation is not necessarily a faithful account of ...`
- `Therefore, explainability should be treated as part of ..., rather than as ...`

### P5｜总体更准不代表各群体同等受益｜约 320 词

1. **Writer claim**：模型不能只凭 overall accuracy 选择，因为预测增益和贷款后果可能在群体间分配不均。
2. **Bono evidence**：80 万名英国借款人、444 个信用特征；extra trees 和 XGBoost 的样本外 AUROC 高于 penalised logistic regression，但切换到 ensemble models 既没有系统性扩大，也没有消除既有公平差异。
3. **Bono implication**：logistic regression 本身也可能有群体误差差异，因此“简单模型天然公平”和“复杂模型天然不公平”都不成立。
4. **Bono limitation**：gender 和 demographic clusters 是代理变量；约 32% 的 thin files 被排除；研究只检查统计公平性，没有观察银行如何把分数转化为价格、准入和福利。
5. **Fuster evidence**：接近 1,000 万笔美国 mortgages；random forest 的 AUC 为 0.8602，对比 nonlinear logistic regression 的 0.8537，average precision 约提高 5.1%，R² 约提高 14.3%。
6. **Fuster distributional result**：Black 与 White Hispanic borrowers 较少获得更低的预测违约概率；简化市场模型中，他们的利率和群体内部利率离散度增加更多。
7. **Fuster limitation**：只观察已经获贷者，存在选择问题；利率和借款行为不是随机的，市场反事实依赖强假设，不能当成精确政策预测。
8. **段尾判断**：总体预测收益必须与 group-specific errors、定价和准入结果一起审计；较低可解释性若同时降低发现和挑战不平等影响的能力，就需要更高而不是更低的收益门槛。

可调用句框：

- `Higher overall accuracy does not imply that ...`
- `This does not show that complex models are inherently unfair; rather, it shows that ...`
- `Accordingly, lenders should evaluate ... instead of inferring fairness from ...`

### Conclusion｜约 200 词

不要增加新文献或新论点，完成四步：

1. **直接回答 extent**：预测增益只在有限、可证明的条件下足以抵偿可解释性下降，不构成普遍采用理由。
2. **合并理由**：增益随数据结构、评价指标和成本设定变化；强可解释模型有时能得到相近表现；事后解释不能保证忠实或未见状态稳定；平均改善还可能掩盖群体差异。
3. **给出模型选择顺序**：先建立强 logistic regression baseline；再测试 spline、scorecard 和 interpretable hybrid；只有黑箱仍有 clear, stable and decision-relevant extra gain 时才进入审计、压力测试和公平性检查。
4. **最终判断**：complex models can be justified as an exception supported by evidence, not as an automatic replacement for logistic regression。

可调用句框：

- `Overall, the evidence supports a limited rather than general justification for ...`
- `The appropriate decision rule is to begin with ..., compare ..., and adopt ... only if ...`
- `Complex models should therefore be treated as ..., not as ...`

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

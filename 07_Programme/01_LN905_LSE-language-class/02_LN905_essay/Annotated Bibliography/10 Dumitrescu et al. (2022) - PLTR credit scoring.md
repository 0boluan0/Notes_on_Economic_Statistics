---
title: Machine learning for credit scoring - Improving logistic regression with non-linear decision-tree effects
authors:
  - Elena Dumitrescu
  - Sullivan Hué
  - Christophe Hurlin
  - Sessi Tokpavi
year: 2022
doi: 10.1016/j.ejor.2021.06.053
status: summary-draft
source_pdf: "[[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/10_Dumitrescu et al. (2022) - Interpretable ML credit scoring (author preprint).pdf]]"
---

# Dumitrescu et al. (2022)：PLTR 混合信用评分模型

> [!cite] Reference
> Dumitrescu, E., Hué, S., Hurlin, C., & Tokpavi, S. (2022). Machine learning for credit scoring: Improving logistic regression with non-linear decision-tree effects. *European Journal of Operational Research, 297*(3), 1178-1192. https://doi.org/10.1016/j.ejor.2021.06.053

> [!summary] 一句话梗概
> 作者提出 penalised logistic tree regression：先用浅层树生成可解释的阈值和交互规则，再由 adaptive-lasso logistic regression 评分；四个数据集显示它显著优于标准逻辑回归，并常与随机森林竞争。

## 研究问题

论文直接挑战“预测性能与可解释性只能二选一”。它问能否保留逻辑回归的透明 link function，同时用决策树自动发现逻辑回归容易遗漏的非线性 threshold 与 interaction effects。

## 方法与证据

- 用变量对训练 short-depth decision trees，从叶节点提取一元和二元 binary rules。
- 把这些规则作为 predictors 放入 adaptive-lasso logistic regression，控制规则数量并保留边际效应。
- 主数据为 Kaggle Give Me Some Credit：150,000 笔贷款、10 个 predictors；另用 Housing、Australian 与 Taiwan 数据稳健性检验。
- 采用 5×2 cross-validation，比较多类模型、五种预测指标、规则复杂度、误分类成本和 expected maximum profit。

## 主要发现

- Kaggle 数据上，PLTR AUC 为 0.8568，略高于随机森林的 0.8529，明显高于线性逻辑回归的 0.6983。
- 四个数据集上 PLTR 均优于标准逻辑回归，并总体与随机森林竞争；Housing 数据上随机森林仍明显更好。
- PLTR 规则最多含两个 predicates，因而比包含大量深层规则的随机森林更容易全局检查。
- 相对线性逻辑回归，PLTR 在 Kaggle 与 Taiwan 数据上的平均误分类成本降幅约为 18.06% 与 22.29%。

## 关键局限

多个 benchmark 数据集较小或较旧，且使用随机 cross-validation 而非时间外推。Kaggle 缺失值用均值填补，可能掩盖数据质量问题。可解释性主要用规则数量和长度衡量，没有让真实信贷人员、客户或监管者做理解测试；若加入更高阶规则提高性能，可解释性也会下降。经济收益还依赖假设的 LGD 与 ROI。

## 对 LN905 essay 的用途

这是最直接挑战题目隐含二元取舍的来源。它支持一个更强的 thesis：在接受可解释性下降之前，应先尝试 hybrid or intrinsically interpretable models；只有当复杂黑箱在这些强基准上仍有稳定、实质优势时，牺牲透明度才可能合理。

## 原文

- [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/10_Dumitrescu et al. (2022) - Interpretable ML credit scoring (author preprint).pdf|Author preprint]]
- [Published article](https://doi.org/10.1016/j.ejor.2021.06.053)


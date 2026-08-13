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

## 成文结构：先拆掉二元取舍，再证明混合方案

> [!abstract] 作者先锁定的中心
> **现象**：行业继续使用可解释的逻辑回归，研究却反复显示随机森林能捕捉其遗漏的阈值和交互。<br>
> **张力**：逻辑回归结构透明但可能误设，随机森林预测强却难以全局说明。<br>
> **中心判断**：问题不必被写成“准确率或可解释性”；可以先用浅树发现非线性规则，再用稀疏逻辑回归组合，从模型设计层同时追求两者。

### Section 路线图

| 原文部分 | 这一部分在全文中的任务 | 它为下一部分打开什么问题 |
|---|---|---|
| 1. Introduction | 建立行业—研究反差，指出黑箱取舍，并预告 PLTR 是 econometrics 与 ML 的组合。 | 逻辑回归具体为什么会输？ |
| 2.1 Non-linear effects in logit | 用门槛和交互的模拟展示线性、二次与手工 interaction logit 的误设。 | 机器学习怎样自动找到这些结构？ |
| 2.2 ML for non-linear effects | 说明树与随机森林如何捕捉阈值，同时量化其规则规模为何难解释。 | 能否只借树来发现规则，而不把最终决策交给森林？ |
| 3.1 PLTR methodology | 从浅层树提取一元/二元 binary rules，再用 adaptive-lasso logit 选择并估计。 | 这个方案在已知真相的环境中是否恢复正确结构？ |
| 3.2 Monte Carlo evidence | 在模拟数据中比较预测、规则数量、长度与边际效应，先验证机制。 | 真实信用数据上是否仍有效？ |
| 4. Benchmark dataset | 介绍 Kaggle 数据与 5×2 CV，用多项统计指标比较七类模型并展示规则。 | 结果是否依赖一个数据集？ |
| 5. Robustness across datasets | 在 Australian、Taiwan、Housing 数据重复比较，暴露 PLTR 并非处处优于 RF。 | 统计提升能否形成经济价值？ |
| 6. Economic evaluation | 用误分类成本和 expected maximum profit 把模型差异转成业务尺度。 | 混合方案最终解决了多少取舍？ |
| 7. Conclusion | 回到行业问题：PLTR 保留 logit link 与边际效应，同时对 RF 有竞争力，但不是无条件统治者。 | — |

### 关键段落组怎样推进

1. **假二分段**：先准确呈现 logit 与 random forest 各自最强优点，说明现有选择为何困难。
2. **根因段**：不把 logit 的失败归结为“太简单”，而定位到 threshold 和 interaction misspecification。
3. **诊断实验段**：在提出新模型前先用 Monte Carlo 证明根因存在；方法因此像答案而不是凭空发明。
4. **设计段**：逐步解释“树负责发现规则、lasso 负责筛选、logit 负责透明评分”，每个部件解决一个问题。
5. **机制验证段**：先在可控模拟中检查是否恢复结构、预测和简洁度。
6. **真实比较段**：在主数据集同时报告多种性能指标与可读规则，不只挑 AUC。
7. **反例与稳健性段**：增加三个数据集，并保留 Housing 上 RF 明显更好的结果，使结论有边界。
8. **经济与结论段**：最后才谈成本和利润，再把结论写成“强可解释替代方案”，而不是“永远最佳模型”。

> [!tip] 可迁移到你的 essay
> 这篇可以成为你整篇文章的结构转折：前半讨论 **black box 是否值得牺牲透明度**，随后用 PLTR 提出更强问题——**在牺牲透明度之前，是否已经测试过能捕捉非线性但仍内生可解释的模型？** 这会让你的 thesis 不停留在折中，而是提出模型选择顺序。

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

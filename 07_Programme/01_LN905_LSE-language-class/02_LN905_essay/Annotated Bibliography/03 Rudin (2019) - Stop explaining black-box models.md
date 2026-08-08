---
title: Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead
authors:
  - Cynthia Rudin
year: 2019
doi: 10.1038/s42256-019-0048-x
status: summary-draft
source_pdf: "[[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/03_Rudin (2019) - Stop explaining black-box models (author preprint).pdf]]"
---

# Rudin (2019)：高风险决策应使用内生可解释模型

> [!cite] Reference
> Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence, 1*, 206-215. https://doi.org/10.1038/s42256-019-0048-x

> [!summary] 一句话梗概
> Rudin 主张，高风险领域不应先部署黑箱再依赖事后解释，因为 surrogate explanation 可能不忠实、误导或被操纵；更安全的默认选择是直接训练内生可解释且性能接近的模型。

## 核心论证

论文反对“复杂度必然带来更高准确率”这一未经验证的前提。对具有清楚、结构化特征的数据，逻辑回归、规则列表或可解释加性模型常能达到与黑箱接近的性能。事后解释是另一个近似模型，只要它比原模型简单，就不可能在所有输入区域完全忠实于黑箱。

## 论证材料

- 区分 inherently interpretable model 与 post-hoc explanation。
- 讨论解释 fidelity、模型调试、错误输入和潜在操纵问题。
- 使用刑事司法、医疗、能源、计算机视觉和信用评分案例。
- 介绍 CORELS、RiskSLIM、稀疏评分系统和 FICO Explainable ML Challenge 等可解释建模方向。

## 主要结论

- 准确率与可解释性之间不存在普遍、必然的取舍；应先用实证比较证明黑箱不可替代。
- 高风险决策中的解释必须忠实于模型实际计算，而不是只与预测大致相关。
- 可解释性具有领域特异性，需要根据使用者、任务和约束设计模型。
- 黑箱可以用于探索或建立性能上限，但不应自动成为最终决策模型。

## 关键局限

这是一篇立场鲜明的理论与评论文章，而非专门针对消费者信用评分的统一实证 benchmark。证据来自跨领域案例，不能量化所有场景下的准确率差异；它也承认可解释模型往往需要更困难的优化和领域设计。

## 对 LN905 essay 的用途

这是反对“少量预测提升即可换取透明度下降”的核心理论来源。最有力的用法不是声称黑箱永远不应使用，而是提出举证责任：在高风险信用决策中，采用者必须先证明复杂模型的提升显著，并证明没有性能相近的内生可解释替代方案。

## 原文

- [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/03_Rudin (2019) - Stop explaining black-box models (author preprint).pdf|Author preprint]]
- [Published article](https://doi.org/10.1038/s42256-019-0048-x)


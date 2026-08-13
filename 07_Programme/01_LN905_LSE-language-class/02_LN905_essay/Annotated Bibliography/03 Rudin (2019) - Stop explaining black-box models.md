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

## 成文结构：先拆前提，再提出替代方案

> [!abstract] 作者先锁定的中心
> **现象**：高风险领域越来越多地使用黑箱；面对透明度批评，主流回应是再训练一个 post-hoc explanation。<br>
> **张力**：解释看似保留了黑箱准确率，却可能不忠实；真正可解释的模型更可信，却被预设为较不准确。<br>
> **中心判断**：在高风险决策中，除非已经证明不存在性能相近的内生可解释模型，否则不应以事后解释为黑箱辩护。

### Section 路线图

| 原文部分 | 这一部分在全文中的任务 | 它为下一部分打开什么问题 |
|---|---|---|
| 1. Introduction | 用司法、医疗等伤害建立 stakes，并严格区分解释黑箱与使用可解释模型。 | 为什么 post-hoc explanation 不能解决问题？ |
| 2. Key Issues with Explainable ML | 依次攻击五个环节：必然 trade-off 的神话、fidelity 不足、解释信息不完整、无法融入数据库外判断、决策流程变复杂。 | 如果问题明显，机构为何仍偏爱黑箱？ |
| 3. Key Issues with Interpretable ML | 承认现实障碍：专有黑箱的利润激励、构造可解释模型的计算与领域成本、对“隐藏模式”的迷信。 | 怎样改变举证责任和治理激励？ |
| 4. Encouraging Responsible ML Governance | 提出两级政策：性能相同则禁用黑箱；至少强制报告可解释模型 benchmark。 | 这些替代模型在技术上是否真的可做？ |
| 5. Algorithmic Challenges in Interpretable ML | 用 optimal logical models、scoring systems、case-based reasoning 展示建设性技术路线。 | 最终原则应如何概括？ |
| 6. Conclusion | 把全文收束为高风险场景的默认规则，而非“所有黑箱永远禁止”的绝对命题。 | — |

### 关键段落组怎样推进

1. **伤害段**：先给实际错误后果，使 interpretability 成为决策安全问题而非审美偏好。
2. **概念切割段**：明确“模型自己可理解”与“另一个模型来近似解释”不是同一件事。
3. **前提反驳段**：先拆最根本的 accuracy-interpretability trade-off；如果这个前提不成立，后续交换就失去依据。
4. **机制反驳段**：再解释 surrogate 为什么必然可能失真，并用案例说明这种失真如何误导人。
5. **让步段**：主动承认可解释模型更难做以及机构为何选择黑箱，避免论证显得天真。
6. **规范转折段**：把困难转化为举证责任——困难不等于可以把风险转嫁给受决策者。
7. **替代方案段**：用三类模型证明立场不是“不要机器学习”，而是“改变模型设计目标”。
8. **限定结论段**：允许黑箱探索性能上限，但拒绝其自动成为最终高风险决策模型。

> [!tip] 可迁移到你的 essay
> 这是写 counterargument 的最好骨架：**先准确陈述流行立场 → 找到它依赖的隐含前提 → 说明失败机制 → 承认对方最强理由 → 提出更窄、更可执行的替代原则**。你可以据此把“预测增益是否 justify”改写成“谁承担证明黑箱不可替代的责任”。

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

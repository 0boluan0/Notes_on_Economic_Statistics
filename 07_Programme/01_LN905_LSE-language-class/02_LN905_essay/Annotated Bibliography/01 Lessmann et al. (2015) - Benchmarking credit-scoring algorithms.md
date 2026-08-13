---
title: Benchmarking state-of-the-art classification algorithms for credit scoring
authors:
  - Stefan Lessmann
  - Bart Baesens
  - Hsin-Vonn Seow
  - Lyn C. Thomas
year: 2015
doi: 10.1016/j.ejor.2015.05.030
status: summary-draft
source_pdf: "[[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/01_Lessmann et al. (2015) - Benchmarking credit-scoring algorithms.pdf]]"
---

# Lessmann et al. (2015)：信用评分算法基准比较

> [!cite] Reference
> Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research, 247*(1), 124-136. https://doi.org/10.1016/j.ejor.2015.05.030

> [!summary] 一句话梗概
> 这篇大规模基准研究表明，多种先进分类器确实能显著优于行业常用的逻辑回归，但“最准确”不必然等于“最有商业价值”，而且结果会受到评价指标、误分类成本和数据环境影响。

## 成文结构：从 benchmark 缺口到条件化结论

> [!abstract] 作者先锁定的中心
> **现象**：信用评分算法快速增加，但许多论文只拿一种新模型和逻辑回归在少量数据上比较。<br>
> **张力**：研究者想宣布“新模型更准”，银行真正需要的却是跨数据、跨指标且能降低实际成本的稳定优势。<br>
> **中心判断**：先进分类器通常优于逻辑回归，但模型价值必须经过广泛 benchmark、统计检验、经济成本和指标敏感性四重检查，不能由单一准确率决定。

### Section 路线图

| 原文部分                               | 这一部分在全文中的任务                                                                  | 它为下一部分打开什么问题         |
| ---------------------------------- | ---------------------------------------------------------------------------- | -------------------- |
| 1. Introduction                    | 先把“算法进步”改写成“现有比较是否可信”的问题，并提出更新 benchmark 的目标。                                | 如果旧证据不够，究竟缺在哪里？      |
| 2. Literature review               | 审查已有研究的算法范围、数据量、评价指标和统计设计，建立新 benchmark 的必要性。                                | 一个公平的新比较必须纳入哪些模型？    |
| 3. Classification algorithms       | 把 41 个方法分为 individual、homogeneous ensemble 和 heterogeneous ensemble，先画清比较空间。 | 怎样保证这些模型在同一实验规则下被评价？ |
| 4. Experimental setup              | 固定 8 个数据集、6 类指标、预处理、重复交叉验证和显著性检验，让后面的排名可比较。                                  | 广泛比较后，谁稳定胜出？         |
| 5.1-5.2 Empirical results          | 先给全体排名，再收窄到 LR、ANN、RF、HCES-Bag 做重点显著性比较。                                     | 统计优势能否转化为实际价值？       |
| 5.3 Financial implications         | 把模型错误放进不同误分类成本比率，测试预测提升是否真的降低业务成本。                                           | 结论会不会只是某个指标的产物？      |
| 5.4 Correspondence across measures | 比较模型排名在不同指标之间是否一致，暴露评价标准依赖性。                                                 | 最终能提出多强、带什么限制的建议？    |
| 6. Conclusions                     | 给出条件化建议：提高 benchmark 标准，同时承认外部有效性与部署成本尚未解决。                                  | —                    |

### 关键段落组怎样推进

1. **现实基准段**：先承认逻辑回归是行业基准，再指出算法研究已经超过旧 benchmark。
2. **文献缺口段**：不是泛称“研究不足”，而是逐项指出模型少、数据少、指标窄、显著性处理弱。
3. **研究设计段**：让每个缺口对应一个设计选择，形成“问题—修复”关系。
4. **广角结果段**：先回答哪些模型整体领先，避免一上来只挑支持作者偏好的模型。
5. **聚焦比较段**：再把候选收窄，检验复杂模型相对逻辑回归的差异是否稳定显著。
6. **价值转换段**：把 accuracy 差异转成误分类成本，回答“提升是否值得”。
7. **反身检查段**：更换指标，看结论是否改写；这一步主动限制自己的结果。
8. **结论段**：不写“最复杂模型最好”，而写“在这些条件下，哪些模型应成为更强 benchmark”。

> [!tip] 可迁移到你的 essay
> 你的“预测收益”section 可以直接借用这个漏斗：**先定义 gain → 检查跨数据与跨指标稳定性 → 转成经济意义 → 最后才判断是否足以交换可解释性**。Lessmann 的价值不只是一个结果，而是提供了判断“material and stable”的顺序。

## 研究问题

作者更新早期信用评分 benchmark，追问三件事：新型分类算法是否稳定优于逻辑回归；不同性能指标是否会改变模型排名；统计准确率的提升能否转化为有管理意义的成本下降。

## 方法与证据

- 在 8 个真实信用评分数据集上比较 41 种分类方法。
- 使用 6 类性能指标，覆盖排序能力、分类正确性和概率校准。
- 通过重复交叉验证和多重比较检验判断差异是否稳健。
- 在不同误分类成本比率下模拟相对逻辑回归的业务成本变化。

## 主要发现

- 多种先进方法显著优于逻辑回归，异质集成方法、随机森林和神经网络整体表现较强。
- 但最新或最复杂的方法并不自动胜出；作者建议未来研究至少把随机森林作为有挑战性的 benchmark，而不能只与逻辑回归比较。
- 相对逻辑回归，神经网络、随机森林和 HCES-Bag 的平均模拟成本改善约为 3.4%、5.7% 和 4.8%。
- 成本结构变化时，统计上最准确的 HCES-Bag 会失去优势，说明准确率与利润并非一一对应。
- AUC、H-measure、KS 和分类正确率给出的模型排序较接近，但 Brier score 与 partial Gini 提供额外信息。

## 关键局限

这是跨数据集的实验 benchmark，不是生产环境部署研究。成本分析是简化模拟，未计入模型治理、解释、维护、数据漂移和监管成本；作者也明确承认实验结果的外部有效性仍有限。研究因此能证明复杂模型“可能有价值”，不能单独证明实际采用一定合理。

## 对 LN905 essay 的用途

这是回答 predictive gains 是否 material and stable 的核心证据。它既能支持复杂模型的正面案例，也能支持你的限定条件：只有当提升跨数据、跨指标并在真实成本函数下持续存在时，降低可解释性才可能被辩护。

## 原文

- [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/01_Lessmann et al. (2015) - Benchmarking credit-scoring algorithms.pdf|Local PDF]]
- [Published article](https://doi.org/10.1016/j.ejor.2015.05.030)

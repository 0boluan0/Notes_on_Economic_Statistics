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


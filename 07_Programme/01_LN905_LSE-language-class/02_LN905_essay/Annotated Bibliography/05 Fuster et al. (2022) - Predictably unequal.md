---
title: Predictably unequal - The effects of machine learning on credit markets
authors:
  - Andreas Fuster
  - Paul Goldsmith-Pinkham
  - Tarun Ramadorai
  - Ansgar Walther
year: 2022
doi: 10.1111/jofi.13090
status: summary-draft
source_pdf: "[[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/05_Fuster et al. (2022) - Predictably unequal.pdf]]"
---

# Fuster et al. (2022)：预测更准，但收益分配不平等

> [!cite] Reference
> Fuster, A., Goldsmith-Pinkham, P., Ramadorai, T., & Walther, A. (2022). Predictably unequal? The effects of machine learning on credit markets. *The Journal of Finance, 77*(1), 5-47. https://doi.org/10.1111/jofi.13090

> [!summary] 一句话梗概
> 机器学习在近千万笔美国抵押贷款上比非线性逻辑回归预测更准，但 Black 与 Hispanic borrowers 获益较少；更精细的风险区分还会扩大部分群体的利率差异和群体内部不确定性。

## 研究问题

论文不仅问机器学习能否提高违约预测，还问预测技术升级产生的 winners and losers 如何按种族与族裔分布，以及差异来自更灵活地拟合可用变量，还是通过其他变量间接推断受限制身份。

## 方法与证据

- 使用接近 1,000 万笔 2009-2013 年发放的美国抵押贷款，并跟踪最长三年的违约结果。
- 比较线性逻辑回归、非线性逻辑回归与 calibrated random forest；XGBoost 作为稳健性检查。
- 建立 flexibility 与 triangulation 两种机制的理论框架。
- 把预测模型嵌入简化的竞争性信贷市场模型，模拟准入和利率的反事实变化。

## 主要发现

- 随机森林的样本外 AUC 为 0.8602，高于非线性逻辑回归的 0.8537；average precision 提升约 5.1%，R² 提升约 14.3%。
- Black 和 White Hispanic borrowers 相对 White non-Hispanic 与 Asian borrowers 更不容易从新模型获得更低的预测违约概率。
- 纯 triangulation 最多解释约 30% 的预测改善；不平等主要来自复杂模型更灵活地利用合法可见特征。
- 简化均衡模型中，总体接受率略升，但 Black 与 Hispanic borrowers 的利率和群体内部利率离散度增加得更多。

## 关键局限

样本只包含已经获得贷款者，未观察被拒者，因此存在选择问题。利率与借款人行为并非随机产生，均衡反事实依赖强假设；作者明确称这些数值不是精确政策预测。研究对象是美国抵押贷款，不能无条件推广到所有消费者信用产品。

## 对 LN905 essay 的用途

这篇文章说明“总体预测增益”不是充分决策标准。即使增益统计显著，模型仍需通过公平性与分配后果审查。它能把你的 thesis 从抽象 interpretability 扩展为具体治理条件：谁从更准确的模型获益，谁承担更精细风险定价的成本。

## 原文

- [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/05_Fuster et al. (2022) - Predictably unequal.pdf|Local PDF]]
- [Published article](https://doi.org/10.1111/jofi.13090)


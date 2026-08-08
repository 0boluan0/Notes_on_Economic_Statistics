---
title: Algorithmic fairness in credit scoring
authors:
  - Teresa Bono
  - Karen Croxson
  - Adam Giles
year: 2021
doi: 10.1093/oxrep/grab020
status: summary-draft
source_pdf: "[[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/07_Bono et al. (2021) - Algorithmic fairness in credit scoring.pdf]]"
---

# Bono et al. (2021)：复杂模型不必然更不公平

> [!cite] Reference
> Bono, T., Croxson, K., & Giles, A. (2021). Algorithmic fairness in credit scoring. *Oxford Review of Economic Policy, 37*(3), 585-617. https://doi.org/10.1093/oxrep/grab020

> [!summary] 一句话梗概
> 在 80 万名英国借款人的信用档案上，ensemble models 比 penalised logit 更准确，却没有系统性恶化已有的群体公平差异；不过它们也没有消除这些问题，且没有任何模型完全满足所有 fairness criteria。

## 研究问题

作者检验从传统逻辑回归切换到 ensemble machine learning 后，整体和各群体预测准确率如何变化，以及 gender、race、health status 与 deprivation 相关群体的 performance parity、separation 和 sufficiency 是否改善或恶化。

## 方法与证据

- 使用 80 万名英国成年人，即约 2% 英国成年人口的信用档案。
- 构造 444 个信用特征；用 2015 年数据训练和测试，再用更晚一期数据模拟部署评价。
- 比较 L2-penalised logistic regression、extremely randomised trees 与 XGBoost。
- 通过称谓推断 gender，并用人口普查地区数据构造 race/health/deprivation clusters。

## 主要发现

- 两个 ensemble models 在总体和主要群体上均比逻辑回归有更高样本外 AUROC。
- 传统模型本身已存在群体间准确率和错误结构差异。
- 切换到机器学习会改善部分指标，但总体上既没有消除、也没有明显扩大检测到的公平问题。
- 直接加入敏感属性并未显著提高违约预测，说明相关信息多已编码在普通信用档案中。

## 关键局限

gender 和 demographic cluster 是代理变量，不是准确的个体身份。研究排除了约 32% 的 thin files，而这些人可能恰是公平风险较高的群体。分析是 observational，只评价分数的统计公平性，不研究银行如何把分数转化为定价、准入和最终福利；不同 fairness criteria 也无法在一般情况下同时完全满足。

## 对 LN905 essay 的用途

它可用来反驳“模型越复杂必然越不公平”的过度推断，并与 [[05 Fuster et al. (2022) - Predictably unequal]] 形成直接对话。两篇合起来支持更精确的结论：公平性不能从模型类型推断，必须在具体数据、指标与最终决策层面单独审计。

## 原文

- [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/07_Bono et al. (2021) - Algorithmic fairness in credit scoring.pdf|Local PDF]]
- [Published article](https://doi.org/10.1093/oxrep/grab020)


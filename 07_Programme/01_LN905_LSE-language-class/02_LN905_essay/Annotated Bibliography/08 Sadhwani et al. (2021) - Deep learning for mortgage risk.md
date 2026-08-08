---
title: Deep learning for mortgage risk
authors:
  - Apaar Sadhwani
  - Kay Giesecke
  - Justin Sirignano
year: 2021
doi: 10.1093/jjfinec/nbaa025
status: summary-draft
source_pdf: "[[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/08_Sadhwani et al. (2021) - Deep learning for mortgage risk (author preprint).pdf]]"
---

# Sadhwani et al. (2021)：超大样本下的抵押贷款深度学习

> [!cite] Reference
> Sadhwani, A., Giesecke, K., & Sirignano, J. (2021). Deep learning for mortgage risk. *Journal of Financial Econometrics, 19*(2), 313-368. https://doi.org/10.1093/jjfinec/nbaa025

> [!summary] 一句话梗概
> 在超过 1.2 亿笔美国抵押贷款和 35 亿条 loan-month observations 上，深度神经网络显著改善多期贷款状态预测，尤其能捕捉 prepayment 与宏观变量之间的强非线性；这是复杂度可能产生实质收益的最强正面案例之一。

## 研究问题

论文研究线性 logistic formulation 是否会误设抵押贷款表现中复杂的非线性与交互关系，以及深度学习能否改善个体贷款、贷款池和投资组合层面的样本外预测。

## 方法与证据

- 数据覆盖 1995-2014 年超过 1.2 亿笔美国 prime 与 subprime mortgages，约占同期发放量的 70%。
- 清理后约 35 亿条月度观察、272 个贷款及宏观变量、7 种贷款状态。
- 训练集截至 2012 年 4 月，validation 为 2012 年 5-10 月，test 为 2012 年 11 月至 2014 年 5 月。
- 比较 0-layer logistic model 与多层 neural networks，并使用 regularisation、dropout 和 ensemble 控制过拟合。

## 主要发现

- 五层网络及其 ensemble 在样本外 likelihood 和多种状态转换 AUC 上优于线性逻辑模型。
- 非线性对 prepayment 尤其重要；失业率与 FICO、LTV、利率和房价存在明显交互。
- state unemployment 在所考察变量中具有最高解释力，说明住房金融与宏观经济联系比许多线性研究呈现得更强。
- 在作者设定的损失假设下，由五层网络选择的 20,000 笔贷款组合一年损失比线性模型组合低 46%；贷款池层面的 prepayment 预测误差也明显更小。

## 关键局限

研究对象是抵押贷款状态与 MBS 风险，不是典型的消费者贷款申请 scorecard。经济收益来自投资组合实验和假设损失率，并非真实部署后的利润。模型可解释性、公平性和监管成本没有直接测量；超大专有数据环境也难以代表一般银行。

## 对 LN905 essay 的用途

它说明当样本极大、关系高度非线性且线性误设会直接影响风险和投资决策时，复杂模型的收益可能足以达到 material threshold。使用时必须限定范围：这是一种“复杂度可能被证明合理”的边界案例，不足以推出普通信用评分也应默认使用深度学习。

## 原文

- [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/08_Sadhwani et al. (2021) - Deep learning for mortgage risk (author preprint).pdf|Author preprint]]
- [Published article](https://doi.org/10.1093/jjfinec/nbaa025)


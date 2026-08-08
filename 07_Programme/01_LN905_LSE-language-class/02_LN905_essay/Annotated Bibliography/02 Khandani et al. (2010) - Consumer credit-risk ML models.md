---
title: Consumer credit-risk models via machine-learning algorithms
authors:
  - Amir E. Khandani
  - Adlar J. Kim
  - Andrew W. Lo
year: 2010
doi: 10.1016/j.jbankfin.2010.06.001
status: summary-draft
source_pdf: "[[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/02_Khandani et al. (2010) - Consumer credit-risk ML models.pdf]]"
---

# Khandani et al. (2010)：消费者信用风险机器学习模型

> [!cite] Reference
> Khandani, A. E., Kim, A. J., & Lo, A. W. (2010). Consumer credit-risk models via machine-learning algorithms. *Journal of Banking & Finance, 34*(11), 2767-2787. https://doi.org/10.1016/j.jbankfin.2010.06.001

> [!summary] 一句话梗概
> 作者把交易流水、账户余额与征信数据结合进 boosted decision-tree 模型，发现模型可提前预测严重逾期，并在一组假设下把预测提升转化为相当于总损失 6%-25% 的潜在节约。

## 研究问题

论文研究高频银行交易信息能否帮助机器学习模型比传统、更新较慢的信用分数更早识别信用卡逾期与违约，以及这些预测是否具有经济和宏观审慎价值。

## 方法与证据

- 使用一家大型商业银行 2005 年 1 月至 2009 年 4 月的专有客户数据。
- 输入包括账户交易、余额、信用局资料和近期财务行为特征。
- 采用带 boosting 的 CART 模型，预测未来 3-12 个月内 90 天以上逾期。
- 通过滚动样本外预测、10-fold cross-validation、信用额度削减模拟和聚合风险时间序列评价模型。

## 主要发现

- 近期收入骤降是明显预警信号：样本中未来六个月严重逾期的无条件概率为 5.3%，而直接存款显著下降者为 10.8%。
- 聚合后的 6 个月和 12 个月逾期预测与实际逾期率的线性回归 R² 约为 85%。
- 在作者设定的保守成本收益假设下，使用预测来削减高风险账户信用额度可节省约总损失的 6%-25%。
- 客户资料越完整，模型预测越好，说明部分增益来自更丰富、更新更及时的数据，而不只是算法本身。

## 关键局限

数据来自单一银行且覆盖金融危机时期，外部有效性有限。经济收益依赖信用额度、利差、违约前余额增长等假设。论文主要与既有信用分数比较，并未在完全相同的输入上建立一个严格的机器学习对逻辑回归实验；也没有处理可解释性、公平性或监管部署成本。

## 对 LN905 essay 的用途

它为“什么叫 material gain”提供直观的货币尺度，是复杂模型一方的重要正面证据。但使用时应强调：论文同时说明数据及时性和信息丰富度是收益来源，因此不能把全部改善都归因于模型复杂度。

## 原文

- [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/02_Khandani et al. (2010) - Consumer credit-risk ML models.pdf|Local PDF]]
- [Published article](https://doi.org/10.1016/j.jbankfin.2010.06.001)


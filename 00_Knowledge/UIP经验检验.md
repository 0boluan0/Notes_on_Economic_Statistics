---
aliases:
  - "短期限样本常拒绝零货币风险溢价的简单 UIP 但结果依赖期限样本和预期测量"
  - Empirical tests of uncovered interest parity
  - UIP and the forward-premium puzzle
  - UIP 经验检验
student_os: knowledge-atom
atom_id: MB-FX-006
atom_set: foreign-exchange-open-economy
atom_type: evidence-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[无抵补利率平价]]"
related:
  - "[[远期升贴水]]"
  - "[[抵补利率平价]]"
part_of:
  - "[[外汇市场、利率平价与不可能三角.canvas]]"
---

# 短期限样本常拒绝零货币风险溢价的简单 UIP 但结果依赖期限样本和预期测量
<!-- bilingual-en:start -->
*Short-horizon samples often reject simple UIP with a zero currency risk premium, but the result depends on horizon, sample, and how expectations are measured*
<!-- bilingual-en:end -->

> [!summary] 原子证据边界
> 经典短期限检验常发现：高利率货币没有按简单 UIP 所预测的幅度贬值，甚至随后升值，使 carry trade 的实现超额回报在样本内可预测。这被称为 UIP puzzle 或 forward-premium puzzle，但它是特定联合假设的经验拒绝，不是“利率与汇率毫无关系”的证明。
> <!-- bilingual-en:start -->
> Classic short-horizon tests often find that high-interest currencies do not depreciate as much as simple UIP predicts and may instead appreciate, making realized carry-trade excess returns predictable in sample. This is called the UIP or forward-premium puzzle. It rejects a particular joint hypothesis; it does not prove that interest rates and exchange rates are unrelated.
> <!-- bilingual-en:end -->

在 CIP 把远期升贴水连接到利差、且使用实现汇率变化代理事前预期时，常见 Fama 回归把未来汇率变化回归于当前远期升贴水或利差。简单 UIP、理性预期和稳定数据生成过程共同预言相应斜率为一；许多经典短期限样本得到很低甚至为负的斜率。
<!-- bilingual-en:start -->
When CIP connects the forward premium to the interest differential and realized depreciation is used as a proxy for the ex ante expectation, the familiar Fama regression relates future exchange-rate changes to the current forward premium or interest differential. Simple UIP, rational expectations, and a stable data-generating process jointly predict a slope of one; many classic short-horizon samples produce a low or even negative slope.
<!-- bilingual-en:end -->

## 一个拒绝对应多种可能机制

- **时变货币风险溢价：** 高利率货币的平均回报可能补偿系统性风险，而不是无成本收益。
- **预期误差和稀有事件：** 实现汇率包含事前不可知的新闻、制度跳变和 peso problem。
- **样本与期限：** 货币、政策区间、结构断点以及一个月或多年期限可能给出不同结果。
- **预期测量：** 用实现汇率、调查预期或模型预期是在检验不同联合假设。
- **市场摩擦：** 交易成本、融资约束与 CIP basis 会改变可交易回报，但不能被一律归为投资者非理性。

因此，“UIP 失败”不等于远期率永远朝反方向预测，也不等于 carry trade 是无风险套利。经验卡只记录检验与识别边界，不提供短期汇率交易信号。
<!-- bilingual-en:start -->
The rejection may reflect time-varying currency risk premia, expectation errors and rare events, sample and horizon dependence, different expectation measures, or market and funding frictions. “UIP fails” therefore does not mean that forward rates always predict in the wrong direction, nor that a carry trade is riskless arbitrage. This card records an empirical and identification boundary, not a short-term trading signal.
<!-- bilingual-en:end -->

> [!question]- 自检
> Fama 回归斜率显著不等于一，能否单独证明投资者预期不理性？
>
> **答案：** 不能。该回归同时依赖零或稳定风险溢价、预期测量、样本稳定性、CIP 连接和误差性质；拒绝的是联合假设。

## 来源与核验

- [Eugene F. Fama, “Forward and Spot Exchange Rates” (1984)](https://doi.org/10.1016/0304-3932(84)90046-1)：核对远期升贴水中预期变化与风险溢价成分的经典识别问题。
- [Federal Reserve IFDP 1068, “Variance Risk Premiums and the Forward Premium Puzzle”](https://www.federalreserve.gov/pubs/ifdp/2012/1068/ifdp1068.htm)：核对经典短期限 UIP 异常与时变货币风险溢价解释。
- [Federal Reserve FEDS 2023-074](https://www.federalreserve.gov/econres/feds/files/2023074pap.pdf)：核对实现汇率与调查预期、风险溢价和预期误差以及子样本依赖的区别。

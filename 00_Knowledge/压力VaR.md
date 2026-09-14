---
aliases:
  - "Basel 2.5 的压力 VaR 用当前组合和相关连续 12 个月重大压力期校准 10 日 99% 单尾损失分位数"
  - "Basel 2.5 压力 VaR"
  - "Basel 2.5 stressed VaR"
  - sVaR
student_os: knowledge-atom
atom_id: RM-VAR-015
atom_set: var-es-backtesting
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
requires:
  - "[[VaR定义]]"
  - "[[风险度量口径]]"
related:
  - "[[VaR时间缩放]]"
  - "[[FRTB 市场风险]]"
  - "[[压力测试方法]]"
  - "[[风险模型验证边界]]"
leads_to:
  - "[[Basel 2.5 VaR资本块]]"
  - "[[压力VaR口径边界]]"
---

# Basel 2.5 的压力 VaR 用当前组合和相关连续 12 个月重大压力期校准 10 日 99% 单尾损失分位数

<!-- bilingual-en:start -->
*Basel 2.5 stressed VaR calibrates the current portfolio's ten-day one-tailed 99% loss quantile to a relevant continuous twelve-month period of significant stress*
<!-- bilingual-en:end -->

> [!summary] 它仍是 VaR，改变的是校准期
> Basel 2.5 的 stressed value-at-risk（sVaR）把银行**当前组合**放在相关历史压力期校准下，计算 **10 日、99% 单尾损失分位点**。模型输入来自经监管认可并定期复核的连续 12 个月重大金融压力期，而不是最近市场窗口。
>
> <!-- bilingual-en:start -->
> Basel 2.5 stressed value-at-risk applies historical stress calibration to the bank's **current portfolio** and computes its **ten-day, one-tailed 99% loss quantile**. Model inputs come from a supervisor-approved and regularly reviewed continuous twelve-month period of significant financial stress rather than the most recent market window.
> <!-- bilingual-en:end -->

## sVaR 固定了什么

设当前组合为 $P_t$。Basel 2.5 要问的是：若与 $P_t$ 有关的市场因子重新处于所选压力期，当前组合的十日损失 99% 分位点是多少？因此必须同时固定四件事：

<!-- bilingual-en:start -->
Let the current portfolio be $P_t$. Basel 2.5 asks for the current portfolio's ten-day 99% loss quantile when its relevant market factors are calibrated to the selected stress period. Four elements must therefore remain fixed:
<!-- bilingual-en:end -->

- **组合：** 使用当前组合，而不是直接报告旧危机时期某个历史组合的损失。
- **统计量：** 仍是 [[VaR定义|VaR 分位点]]，采用 99% 单尾置信水平和十个交易日持有期。
- **校准期：** 模型输入来自连续 12 个月、与当前组合相关的重大金融压力期；这不是任取最近 12 个月，也不是把若干不连续极端日拼接起来。
- **监管约束：** 压力期须获监管认可并定期复核；原标准要求 sVaR 至少每周计算。

<!-- bilingual-en:start -->
- **Portfolio:** use the current portfolio rather than reporting the loss of an old crisis-era portfolio.
- **Statistic:** retain the [[VaR定义|VaR quantile]] at a one-tailed 99% confidence level and a ten-trading-day horizon.
- **Calibration period:** draw model inputs from a relevant continuous twelve-month period of significant financial stress rather than an arbitrary recent year or a collection of disconnected extreme days.
- **Regulatory constraint:** the stress period requires supervisory approval and regular review, and the original standard required sVaR to be calculated at least weekly.
<!-- bilingual-en:end -->

压力期可以因组合而异。2007–2008 年对许多组合可能具有代表性，但标准要求银行考虑对**当前组合**真正相关的压力，而不是机械指定唯一危机年份。

<!-- bilingual-en:start -->
The relevant stress period may differ by portfolio. The 2007–2008 period may be suitable for many portfolios, but the standard requires the bank to consider stress relevant to the **current portfolio** rather than mechanically imposing one crisis interval on every exposure.
<!-- bilingual-en:end -->

## 边界

- sVaR 使用压力期数据，不表示该压力期之后不可能出现更大损失；VaR 本身仍不描述阈值以外的损失严重度。
- 连续 12 个月是校准窗口，不是预测未来压力会持续 12 个月；风险期限仍是十个交易日。
- 压力期获批只确认监管校准窗口，不表示模型的数据、实施、假设和用途已经完成验证。

<!-- bilingual-en:start -->
- Stress-period data do not imply that larger losses are impossible; VaR still does not describe the severity beyond its threshold.
- The continuous twelve months form a calibration window, not a forecast that future stress lasts twelve months; the risk horizon remains ten trading days.
- Approval of the stress period confirms the regulatory calibration window rather than completing validation of the model's data, implementation, assumptions, or use.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> “压力 VaR 使用 2007–2008 年数据，所以它计算的是当年旧组合的危机损失。”这句话错在哪里？
>
> **答案：** sVaR 的对象是当前组合；历史压力期用于校准模型输入，而不是把旧组合本身当作今天的头寸。
>
> <!-- bilingual-en:start -->
> **Self-check:** What is wrong with saying, “Because stressed VaR uses 2007–2008 data, it calculates the crisis loss on the old portfolio”?
>
> **Answer:** The object is the current portfolio. Historical stress data calibrate model inputs; the old portfolio itself does not become today's position set.
> <!-- bilingual-en:end -->

## 来源与核验

- Basel Committee on Banking Supervision, [*Revisions to the Basel II market risk framework*（更新至 2010-12-31）](https://www.bis.org/publ/bcbs193.pdf)，第 718(Lxxvi)(i)–(j) 段：直接核验当前组合、10 日 99% 单尾 VaR、连续 12 个月相关重大压力期、监管认可与复核，以及至少每周计算。

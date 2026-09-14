---
aliases:
  - "Basel 2.5 压力 VaR、一般压力测试与当前 FRTB 压力 ES 是三个不同对象"
  - "压力 VaR、压力测试与 FRTB 压力 ES 的区别"
  - "Stressed VaR, stress testing, and FRTB stressed ES are distinct"
student_os: knowledge-atom
atom_id: RM-VAR-051
atom_set: var-es-backtesting
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
requires:
  - "[[压力VaR]]"
related:
  - "[[Basel 2.5 VaR资本块]]"
  - "[[压力测试方法]]"
  - "[[FRTB 市场风险]]"
---

# Basel 2.5 压力 VaR、一般压力测试与当前 FRTB 压力 ES 是三个不同对象

<!-- bilingual-en:start -->
*Basel 2.5 stressed VaR, general stress testing, and current FRTB stressed ES are three different objects*
<!-- bilingual-en:end -->

> [!summary] 共同使用“压力”不使三者同义
> Basel 2.5 sVaR 是当前组合在历史压力期校准下的十日 99% 损失分位数；一般压力测试把预设冲击或不利结果映射为业务、资本或流动性后果；当前 FRTB 则以压力期校准的 97.5% ES 为核心，并按风险因子流动性期限调整。三者的问题、统计量与监管用途都不同。
>
> <!-- bilingual-en:start -->
> Basel 2.5 sVaR is the current portfolio's ten-day 99% loss quantile under historical stress calibration. General stress testing maps specified shocks or adverse outcomes into business, capital, or liquidity consequences. Current FRTB instead centres on stress-calibrated 97.5% ES adjusted for risk-factor liquidity horizons. Their questions, statistics, and regulatory uses differ.
> <!-- bilingual-en:end -->

| 对象 | 它问什么 | 主要输出与口径 |
|---|---|---|
| Basel 2.5 [[压力VaR]] | 当前组合若相关市场因子重现压力期状态，十日损失分位点是多少？ | 10 日、99% 单尾 VaR；连续 12 个月相关重大压力期 |
| 一般 [[压力测试方法|压力测试]] | 指定敏感度、历史或假设情景，或从重大不利结果反推情景后，会造成什么后果？ | 情景损失、资本、流动性、业务连续性或其他目标量；没有内生固定的置信水平 |
| 当前 [[FRTB 市场风险|FRTB 压力 ES]] | 当前组合在压力校准与监管流动性期限下的尾部平均损失是多少？ | 97.5% 单尾 ES；10 日基础期限、风险因子流动性期限、经批准的缩减风险因子集与最严重 12 个月压力期 |

<!-- bilingual-en:start -->
| Object | Question | Main output and convention |
|---|---|---|
| Basel 2.5 [[压力VaR|stressed VaR]] | What is the current portfolio's ten-day loss quantile if relevant market factors return to stressed conditions? | Ten-day, one-tailed 99% VaR calibrated to a relevant continuous twelve-month period of significant stress |
| General [[压力测试方法|stress testing]] | What follows from specified sensitivities, historical or hypothetical scenarios, or scenarios reverse-engineered from a severe adverse outcome? | Scenario losses, capital, liquidity, business continuity, or another target quantity; no intrinsic fixed confidence level |
| Current [[FRTB 市场风险|FRTB stressed ES]] | What is the current portfolio's average tail loss under stress calibration and regulatory liquidity horizons? | One-tailed 97.5% ES with a ten-day base horizon, risk-factor liquidity horizons, an approved reduced set of risk factors, and the most severe twelve-month stress period |
<!-- bilingual-en:end -->

最容易混淆的地方是“历史危机数据”。sVaR 使用相关压力期校准**当前组合**，并不直接报告危机年份旧组合的损失；历史情景压力测试则可以把一组历史冲击重放到当前暴露上，但其输出不因此自动成为 99% 分位数。FRTB 同样使用压力期，却把目标统计量改为 ES，并另加缩减风险因子集和流动性期限结构。

<!-- bilingual-en:start -->
The main source of confusion is the use of historical crisis data. Stressed VaR calibrates the **current portfolio** to a relevant stress period; it does not report the loss of an old crisis-era portfolio. A historical-scenario stress test may replay historical shocks on current exposures, but that does not make its output a 99% quantile. FRTB also uses a stress period, yet changes the target statistic to ES and adds a reduced risk-factor set and a liquidity-horizon structure.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 某情景重放 2008 年冲击并得到 £40m 损失，能否仅凭“使用压力数据”把 £40m 称为压力 VaR？
>
> **答案：** 不能。还必须有 Basel 2.5 sVaR 所要求的当前组合、十日 99% 分位数和连续 12 个月压力期校准；单个情景损失没有自动获得分位数含义。
>
> <!-- bilingual-en:start -->
> **Self-check:** A scenario replays 2008 shocks and produces a £40m loss. Can £40m be called stressed VaR merely because stress data were used?
>
> **Answer:** No. Basel 2.5 sVaR additionally requires the current portfolio, a ten-day 99% quantile, and calibration to a continuous twelve-month stress period. A single scenario loss does not automatically acquire a quantile interpretation.
> <!-- bilingual-en:end -->

## 边界

- 三者都必须注明规则版本、日期和适用法域；“当前 FRTB”不能反向改写历史 Basel 2.5 课程公式。
- 压力测试可以包含概率模型，但概率解释必须另有依据，不能由“情景很极端”推出。
- FRTB 的压力期 ES 与非可建模风险因子的压力情景资本要求也不是同一个计算对象。

<!-- bilingual-en:start -->
- All three require an explicit rule version, date, and jurisdiction; current FRTB must not be used to rewrite a historical Basel 2.5 course formula.
- A stress test may include a probability model, but any probability interpretation needs separate support and cannot be inferred from the scenario merely being extreme.
- FRTB stress-calibrated ES and the stress-scenario capital requirement for non-modellable risk factors are also distinct calculations.
<!-- bilingual-en:end -->

## 来源与核验

- Basel Committee on Banking Supervision, [*Revisions to the Basel II market risk framework*（更新至 2010-12-31）](https://www.bis.org/publ/bcbs193.pdf)，第 718(Lxxvi)(i)–(j) 段：核验 Basel 2.5 sVaR 的当前组合、10 日 99% 单尾分位数和连续 12 个月相关压力期。
- [Basel Framework, MAR33.3–33.7、33.16–33.17](https://www.bis.org/basel_framework/chapter/MAR/33.htm)：核验当前 FRTB 的 97.5% ES、10 日基础期限、流动性期限、经批准的缩减风险因子集与最严重 12 个月压力期，并区分 NMRF 的压力情景资本要求。
- Basel Committee, [*Stress testing principles*](https://www.bis.org/bcbs/publ/d450.pdf)：核验压力测试框架围绕目标、治理、政策、流程、方法、资源和文档展开，而不是由一个固定分位数定义。

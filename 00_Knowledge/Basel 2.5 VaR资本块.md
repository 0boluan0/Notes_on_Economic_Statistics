---
aliases:
  - "Basel 2.5 的旧市场风险 VaR 资本块把常态 VaR 与压力 VaR 的两个最大值项相加而不是二选一"
  - "Basel 2.5 VaR 与压力 VaR 资本公式"
  - "Basel 2.5 VaR and stressed VaR capital block"
student_os: knowledge-atom
atom_id: RM-VAR-050
atom_set: var-es-backtesting
atom_type: regulatory-formula
status: source-checked
mastery_state: unassessed
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
requires:
  - "[[VaR定义]]"
  - "[[压力VaR]]"
related:
  - "[[VaR回测损益口径]]"
  - "[[压力VaR口径边界]]"
---

# Basel 2.5 的旧市场风险 VaR 资本块把常态 VaR 与压力 VaR 的两个最大值项相加而不是二选一

<!-- bilingual-en:start -->
*The legacy Basel 2.5 market-risk VaR capital block added separate maxima for current-period VaR and stressed VaR rather than choosing between them*
<!-- bilingual-en:end -->

> [!summary] 两个资本项相加
> 忽略增量风险等其他市场风险资本组件时，Basel 2.5 内部模型法同时保留近期市场条件下的 VaR 与压力期校准的 sVaR。每一项先在“最近一次度量”和“乘数乘以此前 60 个营业日均值”之间取较大值，再把两项相加。
>
> <!-- bilingual-en:start -->
> Ignoring other market-risk capital components such as incremental risk, the Basel 2.5 internal-models approach retained both current-period VaR and stress-calibrated sVaR. Each component first takes the larger of its latest measure and a supervisory multiplier times its preceding 60-business-day average; the two components are then added.
> <!-- bilingual-en:end -->

其结构可写成

<!-- bilingual-en:start -->
The structure can be written as
<!-- bilingual-en:end -->

$$
C_t^{\mathrm{VaR\ block}}
=
\max\!\left(VaR_{t-1},\,m_c\overline{VaR}_{60}\right)
+
\max\!\left(sVaR_{t-1},\,m_s\overline{sVaR}_{60}\right).
$$

第一项保留近期市场条件下的 VaR；第二项加入 [[压力VaR|压力期校准的 VaR]]。因此，sVaR 不是在市场平静时才替代普通 VaR 的备用数字，也不是二者择其一。两项各自比较最近值与 60 日均值乘数后才汇总。

<!-- bilingual-en:start -->
The first term retains VaR under recent market conditions; the second adds [[压力VaR|stress-calibrated VaR]]. Stressed VaR is therefore neither a reserve measure that replaces ordinary VaR only in calm markets nor an alternative selected instead of it. Each term is first compared with its own multiplied 60-day average before aggregation.
<!-- bilingual-en:end -->

旧标准把依据事后模型表现设置的 backtesting plus 建立在普通 VaR 的回测结果上，而不是另对 sVaR 运行同一套例外计数。这一事实不能反推 sVaR 无需数据、实施或用途验证；它只说明历史资本公式怎样使用规定的 VaR 回测。

<!-- bilingual-en:start -->
The legacy standard based the backtesting plus tied to ex-post model performance on ordinary VaR rather than on a separate exception count for sVaR. That does not exempt sVaR from validation of data, implementation, or use; it only specifies how the historical capital formula used the prescribed VaR backtest.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 如果普通 VaR 资本项为 12，sVaR 资本项为 18，旧 VaR 资本块是否取两者最大值 18？
>
> **答案：** 不是。忽略其他资本组件时，这两个已经各自完成最大值比较的资本项相加，得到 30。
>
> <!-- bilingual-en:start -->
> **Self-check:** If the ordinary-VaR component is 12 and the sVaR component is 18, does the legacy VaR block take their maximum, 18?
>
> **Answer:** No. Once each component has performed its own maximum comparison, the two are added, giving 30 before other capital components.
> <!-- bilingual-en:end -->

## 边界

- 这是 Basel 2.5 历史内部模型框架中的 VaR 资本块，不是当前 FRTB 的完整市场风险资本公式。
- $m_c$ 与 $m_s$ 是受监管约束的乘数；不能脱离适用规则、日期和法域自行指定。
- 公式中的 60 日是资本计算使用的营业日均值窗口，不是 [[压力VaR]] 的连续 12 个月压力校准期。

<!-- bilingual-en:start -->
- This is the VaR block in the legacy Basel 2.5 internal-models framework, not the complete current FRTB market-risk capital formula.
- $m_c$ and $m_s$ are supervisory multipliers and cannot be chosen without reference to the applicable rule, date, and jurisdiction.
- The 60 days in the formula form a business-day averaging window for capital; they are not the continuous twelve-month stress-calibration period used by [[压力VaR|stressed VaR]].
<!-- bilingual-en:end -->

## 来源与核验

- Basel Committee on Banking Supervision, [*Revisions to the Basel II market risk framework*（更新至 2010-12-31）](https://www.bis.org/publ/bcbs193.pdf)，第 718(Lxxvi)(k)–(l) 段：核验常态 VaR 与 sVaR 两个最大值项相加、各自使用最近值和 60 个营业日均值、乘数设置，以及 backtesting plus 依据普通 VaR 而非 sVaR。

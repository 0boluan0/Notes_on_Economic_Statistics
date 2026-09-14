---
aliases:
  - "PP 用长期方差修正 DF 统计量而不在回归中加入滞后差分"
  - Phillips-Perron test
  - PP test
  - PP 检验
student_os: knowledge-atom
atom_id: TS-UR-014
atom_set: trends-unit-roots-differencing
atom_type: test-principle
status: source-checked
mastery_state: unassessed
requires:
  - "[[DF检验]]"
  - "[[单位根确定项规格]]"
related:
  - "[[ADF检验]]"
  - "[[单位根证据组合]]"
part_of:
  - "[[趋势、单位根与差分.canvas]]"
---

# PP 用长期方差修正 DF 统计量而不在回归中加入滞后差分
<!-- bilingual-en:start -->
*Phillips–Perron corrects DF statistics using long-run variance rather than adding lagged differences*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> Phillips–Perron（PP）检验保留低阶 DF 型回归，但允许误差存在一般的弱依赖和异方差，并用残差的长期方差估计对系数统计量和 $t$ 统计量作非参数修正。它不是“把普通标准误换成稳健标准误”这么简单。

ADF 与 PP 的目标都可针对单位根原假设，但处理短期相关的方式不同：ADF 在回归中显式加入滞后差分；PP 不加入这些滞后项，而通过零频谱密度（长期方差）修正统计量。两者仍必须正确处理截距、趋势与相应的非标准临界值。

PP 需要选择核/截断窗口和带宽；这些选择以及强负 MA 误差可能带来明显的有限样本大小失真。因此 PP 不是自动优于 ADF，也不能把二者同向结果当作独立的“两票”。

> [!question]- 自检
> “PP 与 ADF 完全相同，只是 PP 的标准误更稳健”错在哪里？
>
> **答案：** PP 用长期方差对 DF 的系数与 $t$ 统计量作特定非参数修正；ADF 则把滞后差分直接放入回归。两者的有限样本行为与调节参数也不同。

## 来源与核验

- [Phillips & Perron (1988), *Testing for a Unit Root in Time Series Regression*](https://doi.org/10.1093/biomet/75.2.335)：核对长期方差修正、允许的弱依赖与异方差边界。
- [Schwert (1989), *Tests for Unit Roots: A Monte Carlo Investigation*](https://doi.org/10.3386/t0073)：核对 PP 类统计量在负 MA 误差等设定下可能出现的有限样本大小失真。
- [[02_Economy/01_Econometrics/12_非平稳时间序列.md]]：对照课程 PP 检验段并校正“仅稳健标准误”的过窄说法。

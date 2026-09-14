---
aliases:
  - "KPSS 检验随机趋势创新方差为零且必须区分水平与趋势平稳原假设"
  - KPSS test
  - KPSS
  - KPSS 检验
student_os: knowledge-atom
atom_id: TS-UR-015
atom_set: trends-unit-roots-differencing
atom_type: test-principle
status: source-checked
mastery_state: unassessed
requires:
  - "[[趋势平稳定义]]"
  - "[[随机趋势]]"
related:
  - "[[KPSS统计量]]"
  - "[[单位根证据组合]]"
part_of:
  - "[[趋势、单位根与差分.canvas]]"
---

# KPSS 检验随机趋势创新方差为零且必须区分水平与趋势平稳原假设
<!-- bilingual-en:start -->
*KPSS tests whether the random-trend innovation variance is zero, with distinct level- and trend-stationary nulls*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> KPSS 把序列写成确定性部分、随机游走成分与平稳误差之和，并检验随机游走创新的方差是否为零。方差为零时随机趋势退化为常数，所以原假设是水平平稳或趋势平稳，而不是单位根。

一种表示为
$$
y_t=\delta t+r_t+\varepsilon_t,\qquad
r_t=r_{t-1}+u_t,\quad \operatorname{Var}(u_t)=\sigma_u^2.
$$
KPSS 的原假设是 $H_0:\sigma_u^2=0$。若只从常数回归取残差，检验的是水平平稳；若先回归常数和线性趋势，检验的是围绕该趋势平稳。两者不能混报同一组临界值。

拒绝只说明数据与所选平稳原假设不相容；备择聚焦随机趋势，不等于涵盖每一种非平稳。未拒绝也不是证明平稳，尤其当样本短、长期方差估计不稳定或存在结构突变时。

> [!question]- 自检
> KPSS 未拒绝“趋势平稳”能否推出序列围绕常数水平平稳？
>
> **答案：** 不能。趋势平稳原假设允许非零确定性趋势；水平平稳与趋势平稳是不同规格。

## 来源与核验

- [Kwiatkowski, Phillips, Schmidt & Shin (1992), *Testing the Null Hypothesis of Stationarity against the Alternative of a Unit Root*](https://doi.org/10.1016/0304-4076(92)90104-Y)：核对模型分解、方差分量原假设及水平/趋势规格。
- [[01_Math/06_时间序列分析/lecture.pdf]]：对照课程中的 KPSS 与 ADF 原假设方向。

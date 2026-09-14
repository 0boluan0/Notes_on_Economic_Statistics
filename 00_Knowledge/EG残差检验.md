---
aliases:
  - "Engle-Granger 残差检验以无协整为原假设并使用专用临界值"
  - Engle-Granger residual-test null and critical values
  - EG 残差检验原假设
student_os: knowledge-atom
atom_id: TS-CI-013
atom_set: cointegration-error-correction
atom_type: test-interpretation
status: source-checked
mastery_state: unassessed
requires:
  - "[[Engle-Granger两步法]]"
  - "[[单位根确定项规格]]"
related:
  - "[[Phillips-Ouliaris检验]]"
part_of:
  - "[[协整与误差修正模型.canvas]]"
---

# Engle-Granger 残差检验以无协整为原假设并使用专用临界值
<!-- bilingual-en:start -->
*The Engle-Granger residual test has no cointegration as its null and uses dedicated critical values*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 对第一阶段估计残差 $\hat e_t$ 的检验，原假设是残差含单位根，也就是候选变量不协整；拒绝才支持残差为 $I(0)$。由于 $\hat e_t$ 是用同一数据估出的，统计量不服从普通 DF/ADF 临界分布。

临界值取决于协整回归中的变量个数和确定性项。第一阶段若含截距或趋势，残差检验的参考分布必须与之匹配；不能因为残差样本均值接近零，就随意套“无常数 ADF”表。MacKinnon 的响应面近似常用于给出有限样本临界值或 $p$ 值，但软件默认设置仍需核对。

“未拒绝无协整”只表示当前样本和规格证据不足，不证明绝无长期关系。低功效、错设滞后、结构突变和错误归一化都可能影响结论；报告时应保留统计量、规格、临界值来源和稳健性检查。

> [!question]- 自检
> 第一阶段残差 ADF 的统计量能否与单变量 ADF 表直接比较？
>
> **答案：** 不能。协整向量已经由同一数据估计，必须使用残差型协整检验的专用分布和临界值。

## 来源与核验

- [Engle & Granger (1987)](https://doi.org/10.2307/1913236)：核对残差型检验的原假设和非标准分布。
- [MacKinnon (2010)](https://qed.econ.queensu.ca/working_papers/papers/qed_wp_1227.pdf)：核对协整检验临界值依赖变量数、确定性项与样本量。

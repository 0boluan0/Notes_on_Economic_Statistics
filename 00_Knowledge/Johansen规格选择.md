---
aliases:
  - "Johansen 结论依赖滞后阶数与确定性项规格"
  - Johansen lag and deterministic specification
  - Johansen 规格选择
student_os: knowledge-atom
atom_id: TS-CI-021
atom_set: cointegration-error-correction
atom_type: specification-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Johansen检验]]"
  - "[[单位根确定项规格]]"
related:
  - "[[Johansen秩检验]]"
  - "[[标准协整流程边界]]"
  - "[[协整分析流程]]"
part_of:
  - "[[协整与误差修正模型.canvas]]"
---

# Johansen 结论依赖滞后阶数与确定性项规格
<!-- bilingual-en:start -->
*Johansen conclusions depend on lag order and deterministic-term specification*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> Johansen 的秩结论同时依赖 VAR 滞后阶数与确定性项规格，但它们的作用不同：滞后阶数改变残差化后的样本矩、有效样本量和有限样本表现；常数与线性趋势放在长期空间内还是外，则会改变检验问题及所需的非标准参考分布。季节虚拟变量等平稳控制项也会改变残差化与有限样本结果，但不能由此断言每加一项都必然更换渐近临界值表。

滞后太少会把剩余自相关塞进创新，扭曲秩检验；滞后太多则消耗自由度、降低功效。通常先在水平 VAR 的候选阶数间结合信息准则、残差自相关和系统根选择，再把 $p$ 阶 VAR 转成含 $p-1$ 阶差分的 VECM。标准渐近临界值表不是因滞后数每变一次就换一个新分布族，但滞后会通过规格、有效样本和小样本尺寸／功效影响结论。也不能把软件显示的“VECM lag 1”和“level VAR lag 1”当成同一含义。

确定性项决定检验的是围绕哪种路径的协整：例如，受限常数可表现为协整关系中的截距，非受限常数可在差分中产生漂移；趋势的放置也改变长期关系及参考分布。规格应由数据图形、制度背景和可检验限制共同约束，并做相邻合理规格的敏感性检查。

> [!question]- 自检
> 同一数据在“受限常数”和“非受限常数”下选出不同秩，是否说明软件出错？
>
> **答案：** 不一定。两种规格提出了不同的确定性趋势问题并使用不同参考分布；应解释并比较其合理性。

## 来源与核验

- [Johansen (1991)](https://researchprofiles.ku.dk/en/publications/estimation-and-hypothesis-testing-of-cointegration-vectors-in-gau/)：核对确定性项与高斯 VECM 推断。
- [MacKinnon, Haug & Michelis (1999)](https://doi.org/10.1002/(SICI)1099-1255(199905/06)14:3%3C563::AID-JAE530%3E3.0.CO;2-R)：核对不同确定性规格下 Johansen 检验响应面临界值。

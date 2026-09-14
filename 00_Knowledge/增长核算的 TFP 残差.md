---
aliases:
  - "增长核算把 TFP 作为给定生产函数与投入口径下的残差而非直接观测的纯技术"
  - "Growth accounting treats TFP as a residual under specified production and input measures rather than directly observed pure technology"
  - "Solow residual boundary"
student_os: knowledge-atom
atom_id: MACRO-SOLOW-009
atom_set: solow-growth
atom_type: measurement-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Solow 人均化边界]]"
part_of:
  - "[[Solow 增长模型、稳态与收敛.canvas]]"
related:
  - "[[Solow 的外生技术边界]]"
  - "[[跨国增长回归的因果边界]]"
---

# 增长核算把 TFP 作为给定生产函数与投入口径下的残差而非直接观测的纯技术
<!-- bilingual-en:start -->
*Growth accounting treats TFP as a residual under specified production and input measures rather than directly observed pure technology*
<!-- bilingual-en:end -->

> [!summary] 原子测量
> 对 $Y=ZK^\alpha L^{1-\alpha}$，对数微分给出
> $$g_Z=g_Y-\alpha g_K-(1-\alpha)g_L.$$
> 在 CRS、竞争性要素按边际产出付酬、份额权重适用且投入与产出测量可靠等条件下，$g_Z$ 是不能由已计量资本与劳动增长解释的剩余。它不是直接观测的“纯技术进步”。
> <!-- bilingual-en:start -->
> The Solow residual subtracts weighted measured input growth from output growth. Its interpretation depends on the production function, factor-share assumptions, and measurement of output and inputs; it is not a direct observation of pure technology.
> <!-- bilingual-en:end -->

这里故意用 $Z$ 表示 Hicks-neutral TFP，避免与本组其他原子中的劳动增进技术 $A$ 混淆。对 Cobb–Douglas，
$$
Y=K^\alpha(AL)^{1-\alpha}=A^{1-\alpha}K^\alpha L^{1-\alpha},
$$
所以若写成 Hicks-neutral 形式，则 $Z=A^{1-\alpha}$、$g_Z=(1-\alpha)g_A$。两种记号可以表示同一条技术路径，但其水平与增长率不能直接等同。
<!-- bilingual-en:start -->
Here $Z$ denotes Hicks-neutral TFP, whereas the other Solow atoms use $A$ for labor-augmenting technology. Under Cobb–Douglas, $Z=A^{1-\alpha}$ and therefore $g_Z=(1-\alpha)g_A$; the two technology indexes are related but not numerically interchangeable.
<!-- bilingual-en:end -->

Solow（1957）把生产函数的任何位移都暂称为 technical change，并明确指出教育改善等也会进入该项。实际应用中，劳动质量、资本服务价格、遗漏投入和一般测量误差也可能被残差吸收；若产能利用变化没有进入所测资本服务，它同样会留在剩余中。

因此，增长核算回答的是“在这套核算口径下，已测投入贡献之后还剩多少”，而不是单独识别某项创新的因果效应。若改变资本服务、工时、质量调整或份额权重，残差也会改变。

> [!question]- 自检
> 为什么 $g_Z$ 为正不能单独证明某项技术创新造成了增长？
>
> **答案：** $g_Z$ 是模型与测量条件下的剩余，还可含投入质量、未测资本服务和误差；识别具体技术的因果作用需要额外数据与研究设计。

## 来源与核验

- Solow（1957），[Technical Change and the Aggregate Production Function](https://doi.org/10.2307/1926047)：核对份额加权增长核算式、边际产出付酬假设及“技术变化”作为生产函数位移的广义口径。
- MIT 14.452，[The Solow Growth Model and the Data, Lecture 4](https://ocw.mit.edu/courses/14-452-economic-growth-fall-2016/2617114dcc8bd2a10c633df5b3efa873_MIT14_452F16_Lec4.pdf)，pp. 3–9、41–42：核对增长核算分解、投入质量与资本价格误测，以及 TFP 不必是狭义技术的边界。
- 已从 Cobb–Douglas 生产函数重新做对数微分并把 $g_Z$ 移项为残差；同时区分 Hicks-neutral $Z$ 与劳动增进 $A$ 的记号口径。

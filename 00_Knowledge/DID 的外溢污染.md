---
aliases:
  - "DID 的对照组不能被处理组的政策外溢污染"
  - DID no-interference assumption
  - DID spillover contamination
  - 双重差分无干扰假设
student_os: knowledge-atom
atom_id: ECON-DID-006
atom_type: assumption
status: source-checked
mastery_state: unassessed
part_of:
  - "[[双重差分法（DID）.canvas]]"
requires:
  - "[[双重差分法]]"
leads_to:
  - "[[2×2 DID 识别 ATT]]"
---

# DID 的对照组不能被处理组的政策外溢污染

<!-- bilingual-en:start -->
*A DID comparison group must not be contaminated by spillovers from the treated group*
<!-- bilingual-en:end -->

> [!summary] 核心假设
> 标准 DID 把对照组的实际变化当作“未处理变化”。若处理组的政策会改变对照组的结果，对照组就不再处于真正的未处理状态，第二个差分会同时减掉外溢效应。
>
> <!-- bilingual-en:start -->
> Standard DID treats the comparison group's observed change as an untreated change. If the treated group's policy changes outcomes in the comparison group, that group is no longer genuinely untreated, and the second difference subtracts a spillover effect as well.
> <!-- bilingual-en:end -->

用潜在结果语言说，无干扰要求单位 $i$ 的结果只取决于它自己的处理状态，而不取决于其他单位是否处理。存在地理、市场或网络联系时，更合适的写法往往是 $Y_i(d_i,e_i)$：$d_i$ 是自身处理，$e_i$ 是周围处理暴露。标准四格 DID 把 $e_i$ 省略了，只有在省略它不改变结果或设计已把暴露隔离时才安全。
<!-- bilingual-en:start -->
In potential-outcomes language, no interference means unit $i$'s outcome depends on its own treatment but not on the treatment of other units. With geographic, market, or network links, a more appropriate object may be $Y_i(d_i,e_i)$, where $d_i$ is own treatment and $e_i$ is exposure to others' treatment. The standard four-cell DID omits $e_i$ and is safe only when that omission is harmless or the design isolates exposure.
<!-- bilingual-en:end -->

## 污染会怎样进入估计量

假设某城市提高最低工资，邻市餐馆因此吸引到更多顾客或劳工。若邻市被当作对照组，它的就业变化已经包含政策外溢。此时 DID 估计的是“本市直接变化减去邻市外溢变化”，既不一定等于纯粹直接效应，也不等于政策的总区域效应。外溢为正时，直接效应常被压小；外溢为负时，估计可能被放大，但方向不能脱离具体机制机械判断。
<!-- bilingual-en:start -->
Suppose one city raises its minimum wage and restaurants in a neighbouring city consequently gain customers or workers. If the neighbouring city is used as the comparison group, its employment change already contains a policy spillover. DID then estimates the treated city's change minus the neighbour's spillover, which need not equal either the pure direct effect or the policy's total regional effect. A positive spillover often attenuates the direct-effect contrast, while a negative one may enlarge it, but the direction must be derived from the actual mechanism.
<!-- bilingual-en:end -->

## 不是简单地“换一个远点的对照组”

距离只是暴露的代理。供应链、通勤、价格竞争和信息传播都可能跨越行政边界。研究者应先定义想估计的是直接效应、外溢效应还是总效应，再据此说明哪些单位在每个时期真正可作对照；若干扰本身是研究对象，就需要显式的暴露映射和相应的平行趋势假设。
<!-- bilingual-en:start -->
Distance is only a proxy for exposure. Supply chains, commuting, price competition, and information can cross administrative borders. The researcher should first decide whether the target is a direct, spillover, or total effect, and then justify which units are genuinely valid controls in each period. If interference is itself part of the question, the design needs an explicit exposure mapping and a corresponding parallel-trends condition.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 一项交通政策在处理城市减少拥堵，却把车流赶到对照城市。标准 DID 的对照组变化还能代表“没有政策时处理城市会怎样”吗？
>
> **答案：** 不能直接代表。对照城市已被政策间接处理；估计量会把其外溢变化也减掉，必须重新定义暴露、对照组和目标效应。

## 来源与核验

- Xu (2026), [*Dynamic Difference-in-Differences with Interference*](https://www.aeaweb.org/articles?id=10.1257/pandp.20261108)：核验标准 DID 在干扰下的限制，以及用修正平行趋势识别直接与外溢效应的方向。
- Butts (2021), [*Difference-in-Differences Estimation with Spatial Spillovers*](https://arxiv.org/abs/2105.03737)：核验空间外溢造成的两类偏差——对照组被污染，以及处理组结果同时含自身与邻近处理效应。

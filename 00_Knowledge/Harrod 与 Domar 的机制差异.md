---
aliases:
  - "Harrod 与 Domar 得到相似增长率关系但投资机制和参数含义不同"
  - Harrod and Domar obtain similar growth relations from different mechanisms
  - Harrod Domar distinction
  - Harrod 与 Domar 的模型区别
student_os: knowledge-atom
atom_id: DEV-HD-005
atom_set: harrod-domar-growth
atom_type: historical-distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Harrod—Domar 增长模型.canvas]]"
related:
  - "[[投资双重效应]]"
  - "[[Harrod-Domar 一致增长率]]"
  - "[[Harrod 三种增长率]]"
---

# Harrod 与 Domar 得到相似增长率关系但投资机制和参数含义不同
<!-- bilingual-en:start -->
*Harrod and Domar obtain similar growth-rate relations from different investment mechanisms and parameter meanings*
<!-- bilingual-en:end -->

> [!summary] 原子区分
> “Harrod–Domar”是后来的方便合称，不表示两位作者写了同一个模型。Harrod 从储蓄供给与企业投资需求的相容性定义保证增长率，并区分实际、保证与自然增长；Domar 从投资同时增加收入和产能出发，寻找使两者增长相配、维持充分就业所需的投资增长率。简化后两者可出现同形的增长率关系，但推导问题和参数解释不能无条件互换。
> <!-- bilingual-en:start -->
> The label “Harrod–Domar” joins two historically distinct analyses. Harrod studies compatibility between saving supply and investment demand and distinguishes three growth rates; Domar equates the income and capacity effects of investment to derive the required growth of investment. Their simplified equations can look alike without making the mechanisms identical.
> <!-- bilingual-en:end -->

Harrod 的原式可写为
$$
S_t=sY_t,\qquad I_t=g(Y_{t+1}-Y_t),\qquad S_t=I_t,
$$
所以保证增长率为 $s/g$。这里的 $g$ 描述**投资需求相对于收入增量的系数**；Blume 与 Sargent 特别提醒，它不是 Harrod 原文中固定不变的边际或平均资本生产率，可能随收入、周期和利率变化。把它固定解释成资本—产出比，是后来的教科书重写。

Domar 则令净投资水平 $I$ 以潜在社会平均生产率 $\sigma$ 增加产能，$\dot P=I\sigma$；若边际储蓄倾向为 $s$，乘数侧给出 $\dot Y=\dot I/s$。令产能与收入同步增加，得到
$$
\frac{\dot I}{I}=s\sigma.
$$
只有再把 $\sigma$ 认作某个固定增量资本系数的倒数，并满足相应平衡路径条件，才会与教科书的 $s/v$ 同形。

这里的 $\sigma$ 是全社会潜在产能增量与投资流量之比，不是企业的财务回报率。Domar 还说明，它会受技术、劳动等因素影响；把伴随投资出现的全部产能变化都解释成投资的单独因果效应，也超出了原模型的定义。

因此合并讲授时可以共享一个导航图，却应保留两条来源线：

- **Harrod 线：** 储蓄供给与投资需求相容 → 三种增长率 → 离轨与长期不相容；
- **Domar 线：** 投资双重效应 → 投资与收入必须按能利用新增产能的速度增长。

二者在简化条件下会合于相似的增长要求，但不应把 Harrod 的 natural growth、Domar 的 capacity effect 或两人的参数直接当作彼此原文中的同义词。

> [!question]- 自检
> 为什么两条推导都能写出类似 $s/v$，仍不能说 Harrod 与 Domar 的模型完全相同？
>
> **答案：** 代数形式只显示简化后的均衡条件同形；Harrod 的系数描述投资需求与收入增量，Domar 的 $\sigma$ 描述投资伴随的潜在产能增量。两人的问题、行为机制与参数含义不同。

## 来源与核验

- Harrod（1939），[An Essay in Dynamic Theory](https://doi.org/10.2307/2225181) 与 Domar（1946），[Capital Expansion, Rate of Growth, and Employment](https://doi.org/10.2307/1905364)：分别核对两条原始问题线。
- Blume 与 Sargent（2015），[Harrod 1939](https://doi.org/10.1111/ecoj.12224)：核对 Domar 以不同模型得到相似结果，以及现代教材常把两者合并的历史边界。

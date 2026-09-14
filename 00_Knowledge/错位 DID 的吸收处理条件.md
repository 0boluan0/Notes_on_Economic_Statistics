---
aliases:
  - "标准错位 DID 要求处理可由首次进入时点完整表示"
  - Binary absorbing treatment in staggered DID
  - Irreversible treatment assumption in DID
  - 错位双重差分的处理结构
student_os: knowledge-atom
atom_id: ECON-DID-015
atom_type: boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[双重差分法（DID）.canvas]]"
requires:
  - "[[双重差分法]]"
leads_to:
  - "[[Group-time ATT 聚合]]"
  - "[[DID 估计目标与对照组]]"
---

# 标准错位 DID 要求处理可由首次进入时点完整表示

<!-- bilingual-en:start -->
*Canonical staggered DID requires treatment to be fully represented by the time of first adoption*
<!-- bilingual-en:end -->

> [!summary] 适用边界
> 标准错位 DID 用首次处理时点 $G$ 给单位分组。这只有在处理是二元的、进入后不再退出，而且后续暴露路径确实由 $G$ 决定时才够用。若剂量会变化或处理会反复开关，同一个 $G$ 可以对应多条不同路径，$ATT(g,t)$ 便不再完整描述研究问题。
>
> <!-- bilingual-en:start -->
> Canonical staggered DID groups units by their first treatment date $G$. This is sufficient only when treatment is binary and absorbing and the subsequent exposure path is determined by $G$. If dose varies or treatment switches on and off, the same $G$ can describe different paths, so $ATT(g,t)$ no longer fully represents the causal question.
> <!-- bilingual-en:end -->

在 Callaway 与 Sant’Anna 的基准框架中，$D_{it}$ 只能从 0 变成 1；一旦 $D_{i,t-1}=1$，下一期仍有 $D_{it}=1$。因此，首次进入时间 $G_i$ 足以编码单位的整条处理路径，潜在结果可以写成“若在 $g$ 期首次处理”的 $Y_{it}(g)$。这不是所有 DID 的普遍事实，而是该组方法成立所依赖的处理结构。
<!-- bilingual-en:start -->
In Callaway and Sant’Anna’s baseline framework, $D_{it}$ can move only from 0 to 1: once $D_{i,t-1}=1$, the unit remains treated in the next period. The first adoption date $G_i$ therefore encodes the entire treatment path, and potential outcomes can be indexed as $Y_{it}(g)$, the outcome under first treatment in period $g$. This is not a universal property of DID; it is part of the treatment structure on which this class of methods relies.
<!-- bilingual-en:end -->

## 连续剂量不能被首次进入时间代替

若补贴强度为 $d$，研究者要先说明比较的是 $d$ 与 0、两个正剂量之间的差，还是剂量边际效应，并据此定义 $Y_t(d)$。广义平行趋势或许能识别各剂量相对于零剂量的“该剂量接受者效应”，却不会自动消除不同单位选择不同剂量造成的跨剂量选择。把剂量线性放进 TWFE 还会额外施加函数形式与权重解释。
<!-- bilingual-en:start -->
With treatment intensity $d$, the researcher must specify whether the target compares $d$ with zero, compares two positive doses, or concerns a marginal dose effect, and define $Y_t(d)$ accordingly. A generalized parallel-trends condition may identify treated-on-the-treated effects relative to zero dose, but it does not automatically remove selection across positive doses. Entering dose linearly in TWFE also imposes additional functional-form and weighting assumptions.
<!-- bilingual-en:end -->

## 可逆处理需要整条历史

若单位会退出处理或反复切换，今天的 $D_{it}=0$ 既可能表示从未处理，也可能表示已经退出。滞后、累积或退出效应会使这两种状态产生不同结果。此时应以完整处理历史 $\bar D_{it}=(D_{i1},\ldots,D_{it})$ 定义潜在结果，或采用明确允许处理切换（switching）的框架；仅按首次处理时间划分处理批次（cohort）会把不同暴露路径混在一起。
<!-- bilingual-en:start -->
When units can leave treatment or switch repeatedly, $D_{it}=0$ today may mean either never treated or previously treated and now off treatment. Lagged, cumulative, or withdrawal effects make those states different. Potential outcomes must then be indexed by the treatment history $\bar D_{it}=(D_{i1},\ldots,D_{it})$, or the analysis must use a framework that explicitly permits switching. Grouping only by first adoption mixes distinct exposure paths.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> A 组在 2025 年首次领取补贴后连续领取三年，B 组也在 2025 年首次领取，但第二年退出、第三年重新加入。为什么只用 $G=2025$ 不能完整描述二者的处理？
>
> **答案：** 两组首次进入时间相同，后续处理路径却不同；若效应会累积、滞后或因退出而改变，它们在同一 $ATT(g,t)$ 中不再代表同一种暴露。

## 来源与核验

- Callaway & Sant’Anna (2021), [*Difference-in-Differences with Multiple Time Periods*](https://psantanna.com/files/Callaway_SantAnna_2020.pdf)，Assumption 1 与第 2.1 节：核验二元、不可逆处理以及首次处理时点 $G$ 能编码处理路径的基准设定。
- Callaway, Goodman-Bacon & Sant’Anna (2024), [*Difference-in-Differences with a Continuous Treatment*](https://www.nber.org/papers/w32117)：核验连续处理下的剂量特定参数、跨剂量选择问题，以及线性 TWFE 系数解释的额外限制。

---
aliases:
  - "在标准 Solow 条件下资本从两侧趋向稳态但参数变化后的整条路径不是比较静态"
  - "Under standard Solow conditions capital converges from either side while the adjustment path is not comparative statics"
  - "Solow transition dynamics"
student_os: knowledge-atom
atom_id: MACRO-SOLOW-004
atom_set: solow-growth
atom_type: dynamic-mechanism-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Solow 稳态条件]]"
part_of:
  - "[[Solow 增长模型、稳态与收敛.canvas]]"
implies:
  - "[[Solow 收敛边界]]"
  - "[[储蓄率的水平效应]]"
related:
  - "[[Harrod 离轨不稳定性]]"
---

# 在标准 Solow 条件下资本从两侧趋向稳态但参数变化后的整条路径不是比较静态
<!-- bilingual-en:start -->
*Under standard Solow conditions capital converges from either side, while the path after a parameter change is not comparative statics*
<!-- bilingual-en:end -->

> [!summary] 原子动态
> 固定 $s,n,g,\delta$ 并满足标准生产函数条件时，若 $k<k^*$，则 $sf(k)>(n+g+\delta)k$、$\dot k>0$；若 $k>k^*$，不等号反向、$\dot k<0$。这给出模型内向稳态的转型动态。比较静态只比较参数变化前后的 $k^*$，而 comparative dynamics 还追踪资本、产出、消费和增长率怎样随时间调整。
> <!-- bilingual-en:start -->
> With fixed parameters and the standard Solow restrictions, net investment is positive below $k^*$ and negative above it, so capital moves toward the steady state. Comparative statics compare old and new endpoints; transition or comparative dynamics describe the entire adjustment path after a shock.
> <!-- bilingual-en:end -->

例如储蓄率永久上升时，投资曲线立即上移，但既有资本存量不能瞬间跳到新稳态。经济先从旧 $k^*$ 出发，经历正的净投资、资本加深和暂时较快的增长，再逐渐接近更高的新稳态。更高人口增长或折旧则抬高 break-even 线，使目标稳态降低，路径方向相反。

“全局稳定”是给定这组方程与标准条件的模型结论，不是现实经济必然稳定。非凹技术、多个稳态、内生储蓄、调整成本、冲击或制度变化都会改变路径；也不能由一张相图推断实际调整速度。

> [!question]- 自检
> 为什么“储蓄率上升使 $k^*$ 上升”还没有回答过渡期发生什么？
>
> **答案：** 那只是新旧终点的比较静态；还需从既有资本出发，用运动方程追踪净投资、资本、产出和增长率怎样逐期变化。

## 来源与核验

- [[02_Economy/10_发展经济学/发展经济学拍屏ppt.pdf#page=72|发展经济学课程 PDF pp. 72、75–77]]：核对相图两侧的运动方向、储蓄率冲击与人口增长冲击。
- MIT 14.452，[The Solow Growth Model, Lectures 2–3](https://ocw.mit.edu/courses/14-452-economic-growth-fall-2016/2b68057aa4e74410d00ae89a0c49752f_MIT14_452F16_Lec2and3.pdf)，pp. 35–44、81–84：核对标准条件下的单调收敛、渐近稳定与 comparative dynamics 的定义。

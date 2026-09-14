---
aliases:
  - "Hotelling T² 等于所有非零线性投影 t² 的最大值"
  - Hotelling T-squared is the maximum projected t-squared
  - Hotelling T² 的最大投影解释
  - 多元全局检验与逐坐标 t 检验的区别
student_os: knowledge-atom
atom_id: STAT-HOT-002
atom_set: hotelling-mean-inference
atom_type: equivalence-interpretation
status: source-checked
mastery_state: unassessed
requires:
  - "[[单样本Hotelling T²]]"
  - "[[正定与半正定可逆性]]"
part_of:
  - "[[Hotelling T² 与多元均值推断.canvas]]"
implies:
  - "[[Hotelling置信椭球]]"
related:
  - "[[Bonferroni对比区间]]"
---

# Hotelling T² 等于所有非零线性投影 t² 的最大值
<!-- bilingual-en:start -->
*Hotelling's T-squared is the maximum squared t-statistic over all nonzero linear projections*
<!-- bilingual-en:end -->

> [!summary] 原子等价
> 令 $d=\bar X-\mu_0$ 且 $S\succ0$。对任意非零方向 $a\in\mathbb R^p$，定义投影的一元统计量
> $$t^2(a)=\frac{n(a^Td)^2}{a^TSa}.$$
> 则
> $$T^2=n d^TS^{-1}d=\sup_{a\ne0}t^2(a).$$
> 当 $d\ne0$ 时，最大方向满足 $a\propto S^{-1}d$。
> <!-- bilingual-en:start -->
> The omnibus statistic searches every linear projection and takes the largest squared standardized mean departure; it is not a collection of coordinatewise tests.
> <!-- bilingual-en:end -->

证明只需在 $S$ 诱导的几何里用 Cauchy–Schwarz：
$$
(a^Td)^2
=\bigl((S^{1/2}a)^T(S^{-1/2}d)\bigr)^2
\le(a^TSa)(d^TS^{-1}d),
$$
取 $a=S^{-1}d$ 达到等号。

因此全局拒绝 $H_0:\mu=\mu_0$ 只说明**至少有一个线性组合**偏离目标；它不说明每个坐标都显著。分别做 $p$ 个 t 检验只搜索 $a=e_1,\ldots,e_p$ 这些坐标方向；若不做多重性校正还会膨胀家族错误率，即使做了校正，它回答的仍只是所选坐标组，而不是所有协方差加权方向构成的整体检验。

例如两个响应高度正相关时，方向 $(1,-1)$ 的自然变异可能很小。两个边际偏离各自都不大，但它们的差沿低方差方向很异常，$T^2$ 仍可很大。

> [!warning] 边界
> 若 $d=0$，最大值为 0，最大方向不唯一。若 $S$ 奇异，分母可能在某些非零方向为 0，上述普通逆与有限最大值等价式失去前提；不能用这条公式为经典 F 校准“补上”一个伪逆版本。

> [!question]- 自检
> 全局 $T^2$ 显著时，为什么不能写成“所有响应均值都显著不同”？
>
> **答案：** $T^2$ 取所有线性投影中最大的标准化偏离；一次拒绝只保证至少一个方向提供反证，可能是单一坐标，也可能是多个坐标的组合。

## 来源与核验

- [[01_Math/04_多元统计分析/05_ 总体平均向量的推论.md#1.2.2. 从一元 t 检验到多元 T²|多元统计课程 §1.2.2]]：核对协方差加权二次型及其一元 t 类比。
- [Stanford STATS 305C, *Hotelling T2*, “Largest t statistic interpretation”](https://web.stanford.edu/class/stats305c/lectures/Hotelling_T2.html#largest-t-statistic-interpretation)：核对 $T^2=\sup_a t^2(a)$、最大方向与两样本同构。
- [Penn State STAT 505, Lesson 7](https://online.stat.psu.edu/stat505/Lesson07)：核对向量原假设与后续分量定位是不同推断问题。

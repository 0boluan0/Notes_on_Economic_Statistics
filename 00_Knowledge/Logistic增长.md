---
aliases:
  - "Logistic 方程让人均增长率随规模线性下降并使正初值轨迹趋向承载量"
  - "The logistic equation makes per-capita growth fall linearly with population and drives positive trajectories toward carrying capacity"
  - Logistic growth
student_os: knowledge-atom
atom_id: CALC-APP-030
atom_set: derivative-applications
atom_type: model
status: source-checked
mastery_state: unassessed
part_of:
  - "[[导数的应用.canvas]]"
requires:
  - "[[指数增长与衰减]]"
  - "[[分离变量]]"
related:
  - "[[稳态与稳定性]]"
  - "[[微分方程最大区间]]"
---

# Logistic 方程让人均增长率随规模线性下降并使正初值轨迹趋向承载量
<!-- bilingual-en:start -->
*The logistic equation makes per-capita growth fall linearly with population and drives positive trajectories toward carrying capacity*
<!-- bilingual-en:end -->

> [!summary] 原子模型
> 对 $r,K>0$，
> $$
> y'=ry\left(1-\frac{y}{K}\right)
> $$
> 有平衡解 $y=0,K$。若 $y(t_0)=y_0>0$，则非平衡解可写为
> $$
> y(t)=\frac{K}{1+\left(\frac{K-y_0}{y_0}\right)e^{-r(t-t_0)}}.
> $$
> 该式取包含 $t_0$ 且分母不为零的最大区间；沿正时间方向，$K$ 是正半轴上的吸引承载量。
> <!-- bilingual-en:start -->
> For $r,K>0$, the logistic equation $y'=ry(1-y/K)$ has equilibria zero and $K$. A positive non-equilibrium initial value gives $y(t)=K/[1+((K-y_0)/y_0)e^{-r(t-t_0)}]$ on the maximal interval containing $t_0$ where its denominator is nonzero. In forward time, positive trajectories approach the carrying capacity $K$.
> <!-- bilingual-en:end -->

不求解也能先读方向：$0<y<K$ 时 $y'>0$，$y>K$ 时 $y'<0$。除以 $y(K-y)$ 前必须保留两个平衡解。课程中的形式 $y'=\rho y(s-y)$ 与本卡相同，只需令 $K=s$、$r=\rho s$。
<!-- bilingual-en:start -->
The phase direction is visible before solving: growth is positive between zero and $K$ and negative above $K$. Both equilibria must be recorded before dividing by $y(K-y)$. The course form $y'=\rho y(s-y)$ is identical after setting $K=s$ and $r=\rho s$.
<!-- bilingual-en:end -->

Logistic 不是“所有受限增长”的同义词。它假设人均增长率 $y'/y=r(1-y/K)$ 对规模线性下降，参数恒定且没有时滞、年龄结构或随机冲击。S 形曲线本身不足以识别这一机制；$K$ 也是模型参数，不必等于环境中永恒不变的物理上限。
<!-- bilingual-en:start -->
Logistic growth is not a synonym for every bounded process. It assumes a linearly declining per-capita rate, constant parameters, and no delay, age structure, or random shocks. An S-shaped curve alone does not identify this mechanism, and $K$ is a model parameter rather than an immutable physical ceiling.
<!-- bilingual-en:end -->

“正轨迹趋向 $K$”是向前时间的陈述，不等于解对全部实数都有定义。当 $y_0>K$ 时，显式式的分母会在某个更早时刻变为零，向后延拓在那里爆破；必须按 [[微分方程最大区间]] 限制解。
<!-- bilingual-en:start -->
Convergence to $K$ is a forward-time statement, not a claim of existence for every real time. When $y_0>K$, the denominator vanishes at an earlier time and the backward continuation blows up there, so the solution must be restricted to its maximal interval.
<!-- bilingual-en:end -->

> [!question]- 自检
> 当 $y=K/2$ 时，为什么总增长率最大，但人均增长率不是最大？
>
> **答案：** 总增长 $ry(1-y/K)$ 是关于 $y$ 的开口向下二次式，在 $K/2$ 达峰；人均增长率 $r(1-y/K)$ 随 $y$ 单调下降，在 $y\to0^+$ 时趋向上确界 $r$，但在正规模域内不取得它。
> <!-- bilingual-en:start -->
> **Check:** Why is total growth highest at $y=K/2$, while per-capita growth is not?
>
> **Answer:** Total growth $ry(1-y/K)$ is a downward-opening quadratic in $y$, with its peak at $K/2$. The per-capita rate $r(1-y/K)$ decreases with $y$ and approaches its supremum $r$ as $y\to0^+$, without attaining it at a positive population.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise040_Solutions.pdf#page=1|MIT 18.01SC Exercise 40 solution]]：核对 $y'=\rho y(s-y)$ 的符号、分离变量解及趋向 $s$。
- [[01_Math/01_calculus/02_Applications_of_Differentiation#配套练习：指数与受限增长|课程记录受限增长练习]]：核对平衡解与参数化的完整写法。

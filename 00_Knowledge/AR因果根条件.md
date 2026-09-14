---
aliases:
  - "AR 因果性要求滞后多项式根在单位圆外而伴随矩阵特征值在单位圆内"
  - AR stationarity roots
  - AR causality condition
  - Causal AR(1) stationarity
  - 自回归稳定根条件
student_os: knowledge-atom
atom_id: TS-ARMA-006
atom_set: arma-modeling
atom_type: criterion
status: source-checked
mastery_state: unassessed
requires:
  - "[[AR(p)模型]]"
  - "[[滞后算子约定]]"
related:
  - "[[离散系统谱稳定性]]"
  - "[[ARMA无限MA表示]]"
  - "[[递推根与滞后零点]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
  - "[[差分方程与滞后算子.canvas]]"
---

# AR 因果性要求滞后多项式根在单位圆外而伴随矩阵特征值在单位圆内
<!-- bilingual-en:start -->
*AR causality puts lag-polynomial roots outside the unit circle and companion eigenvalues inside it*
<!-- bilingual-en:end -->

> [!summary] 原子判据
> 对
> $$\phi(z)=1-\phi_1z-\cdots-\phi_pz^p,$$
> 由当前与过去创新生成的稳定因果解要求 $\phi(z)=0$ 的每个零点满足 $|z|>1$。同一递推的 companion matrix 特征值 $\lambda$ 是这些零点的倒数，所以要求 $|\lambda|<1$。
> <!-- bilingual-en:start -->
> For the stated lag-polynomial convention, a stable causal solution requires every zero of $\phi(z)$ to lie outside the unit circle. The companion-matrix eigenvalues are their reciprocals and therefore lie inside the unit circle.
> <!-- bilingual-en:end -->

AR(1) 是最小核对：$\phi(z)=1-\phi z$ 的零点是 $z=1/\phi$，而递推特征值就是 $\phi$。所以“多项式根在外”与“递推特征值在内”都等价于 $|\phi|<1$，二者没有矛盾，只是对象不同。

这个条件给出单边表示 $y_t-\mu=\sum_{j\ge0}\psi_j\varepsilon_{t-j}$，使冲击权重衰减。更细的边界是：若 AR 根不在单位圆上但有根位于圆内，数学上可能构造依赖**未来**创新的非因果平稳解；它不符合通常的预测型 ARMA 解释。因此课程语境下应说“因果平稳 ARMA 的根在外”，而不是把所有可能的平稳解都混为一谈。
<!-- bilingual-en:start -->
For AR(1), the lag-polynomial zero $1/\phi$ lies outside exactly when the recursion eigenvalue $\phi$ lies inside. The root condition yields a one-sided expansion in current and past innovations. A noncausal stationary solution may exist when roots avoid the unit circle but some lie inside; it depends on future innovations and is outside the usual forecasting interpretation.
<!-- bilingual-en:end -->

> [!question]- 自检
> $1-1.25B$ 的零点是 $0.8$。它能否给出通常的因果平稳 AR(1)？
>
> **答案：** 不能。滞后多项式零点在单位圆内；等价地递推系数/伴随特征值 $1.25$ 在单位圆外。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=87|课程讲义 pp. 87–96]]：核对 AR(1)、AR($p$) 齐次递推与 MA($\infty$) 展开的课程约定。
- [Hyndman & Athanasopoulos, FPP3 §9.3](https://otexts.com/fpp3/AR.html)：核对 AR 平稳参数区间与特征根条件。
- [MIT OCW 18.S096, Lecture 8](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/1926c83ecd7ea700f7cb63914c6d7c0f_MIT18_S096F13_lecnote8.pdf)：核对因果展开与单位圆根条件。

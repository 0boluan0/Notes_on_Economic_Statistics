---
aliases:
  - "稳定 VAR 的无条件协方差解离散 Lyapunov 方程"
  - VAR unconditional covariance
  - Discrete Lyapunov equation for VAR
student_os: knowledge-atom
atom_id: TS-VAR-009
atom_set: vector-autoregression
atom_type: derivation
status: source-checked
mastery_state: unassessed
requires:
  - "[[VAR因果VMA表示]]"
  - "[[VAR伴随形式]]"
related:
  - "[[VAR多步预测误差]]"
  - "[[矩阵幂趋零判据]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# 稳定 VAR 的无条件协方差解离散 Lyapunov 方程
<!-- bilingual-en:start -->
*The unconditional covariance of a stable VAR solves a discrete Lyapunov equation*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 稳定 VAR 的同期无条件协方差是所有滞后创新传播方差的和；在伴随状态中，它等价于离散 Lyapunov 方程的唯一解，而不是把长期乘数夹在创新协方差两侧。

先看去均值 VAR(1)：
$$
y_t=A y_{t-1}+u_t,\qquad \operatorname{Cov}(u_t)=\Sigma_u.
$$
若 $u_t$ 与过去正交，令 $\Omega=\operatorname{Cov}(y_t)$，则
$$
\Omega=A\Omega A'+\Sigma_u
=\sum_{i=0}^{\infty}A^i\Sigma_u(A')^i.
$$
当 $\rho(A)<1$ 时，这个离散 Lyapunov 方程有唯一的有限半正定解。向量化后，
$$
\operatorname{vec}(\Omega)
=\left(I-A\otimes A\right)^{-1}\operatorname{vec}(\Sigma_u).
$$

对 VAR($p$)，在伴随状态 $Y_t=FY_{t-1}+U_t$ 中令
$$
Q=\operatorname{Cov}(U_t)
=\begin{pmatrix}
\Sigma_u&0\\0&0
\end{pmatrix}.
$$
状态协方差 $\Gamma$ 解
$$
\Gamma=F\Gamma F'+Q,
$$
原变量协方差是其左上 $K\times K$ 块，即 $J\Gamma J'$。

一般而言，
$$
(I-A)^{-1}\Sigma_u(I-A')^{-1}
$$
不是 $y_t$ 的同期无条件协方差；它用的是长期累计乘数并包含不同滞后之间的交叉项。只有在特殊退化条件下两者才会偶然相同，不能把它当作通用公式。

> [!question]- 自检
> 稳定 VAR(1) 的 $\Omega$ 为什么不是只取 $\Sigma_u+A\Sigma_u A'$？
>
> **答案：** 因为更早创新仍会经 $A^2,A^3,\ldots$ 传播到当前值；Lyapunov 解把所有这些期限的方差贡献都加总。

## 来源与核验

- [Lütkepohl (2005), *New Introduction to Multiple Time Series Analysis*](https://doi.org/10.1007/978-3-540-27752-1)，第 2 章：核对稳定 VAR 的协方差矩阵与伴随表示。
- [[矩阵幂趋零判据]]：核对 Lyapunov 级数收敛所需的稳定条件。
- [[01_Math/06_时间序列分析/lecture.pdf]]：对照课程公式，并纠正把长期乘数公式误作同期协方差的写法。

---
aliases:
  - "每个离散时间次鞅都唯一分解为鞅与从零开始的可预测递增过程之和"
  - Doob decomposition
  - Doob decomposition theorem
student_os: knowledge-atom
atom_id: PROB-MG-007
atom_set: martingales-stopping
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[次鞅]]"
  - "[[离散可预测过程]]"
related:
  - "[[鞅定义]]"
  - "[[鞅差序列]]"
  - "[[鞅差部分和]]"
part_of:
  - "[[鞅与停时.canvas]]"
---

# 每个离散时间次鞅都唯一分解为鞅与从零开始的可预测递增过程之和
<!-- bilingual-en:start -->
*Every discrete-time submartingale decomposes uniquely into a martingale plus a predictable increasing process starting from zero*
<!-- bilingual-en:end -->

> [!summary] 定理
> 设 $X=(X_n)_{n\ge0}$ 是相对于滤过 $(\mathcal F_n)$ 的离散时间[[次鞅]]。则存在唯一的过程 $M$ 与 $A$，使
> $$
> X_n=M_n+A_n,
> $$
> 其中 $M$ 是鞅，$A$ 可积，$A_0=0$，$A$ 几乎处处递增，并且对每个 $n\ge1$，$A_n$ 对 $\mathcal F_{n-1}$ 可测，即 $A$ 是[[离散可预测过程|可预测过程]]。

次鞅按定义已经要求每个 $X_n$ 可积；离散时间 Doob 分解不需要额外假设平方可积。

## 分解的构造

把每一步由过去信息决定的条件均值增量收进 $A$：
$$
A_0=0,
\qquad
A_n-A_{n-1}
=E(X_n\mid\mathcal F_{n-1})-X_{n-1}\ge0.
$$
因此
$$
A_n=\sum_{k=1}^n
\left[E(X_k\mid\mathcal F_{k-1})-X_{k-1}\right],
\qquad
M_n=X_n-A_n.
$$
每个 $A_n-A_{n-1}$ 都对 $\mathcal F_{n-1}$ 可测、可积且非负，所以 $A$ 可预测、可积并递增；$M=X-A$ 也适应且可积。另一方面，
$$
E(M_n\mid\mathcal F_{n-1})
=E(X_n\mid\mathcal F_{n-1})-A_n
=X_{n-1}-A_{n-1}
=M_{n-1},
$$
故 $M$ 是鞅。

## 规范化保证唯一性

若不固定 $A_0$，常数可以在 $M$ 与 $A$ 之间移动。规定 $A_0=0$ 后，鞅的条件零均值增量强制给出
$$
A_n-A_{n-1}
=E(X_n-X_{n-1}\mid\mathcal F_{n-1})
=E(X_n\mid\mathcal F_{n-1})-X_{n-1}.
$$
所以 $A$ 被逐期唯一确定，$M=X-A$ 也随之唯一。

## 分解在区分什么

- $A$ 收集上一期信息已经能确定的非负条件均值增量；
- $M$ 保留给定过去后条件均值为零的创新，其逐期增量构成[[鞅差序列]]。

这是一种相对于所选滤过的概率分解，而不是因果分解。改变信息流会改变条件期望，也会改变 $A$ 与 $M$。若 $X$ 本身已是鞅，则每个 $A_n-A_{n-1}=0$，所以 $A\equiv0$。

> [!question]- 自检
> 怎样从一步条件期望直接读出 Doob 分解中的可预测增量？
>
> **答案：** 取
> $$\Delta A_n=E(X_n\mid\mathcal F_{n-1})-X_{n-1}.$$
> 次鞅条件保证它非负；它对 $\mathcal F_{n-1}$ 可测，因此是可预测增量。再令 $M_n=X_n-A_n$。

## 来源与核验

- [MIT OCW 15.070J, Lecture 10, Theorem 1, pp. 1–2](https://ocw.mit.edu/courses/15-070j-advanced-stochastic-processes-fall-2013/36fffa59f09fbedd5969539c90df4cbb_MIT15_070JF13_Lec10.pdf#page=2)：核对离散时间 Doob 分解的存在、唯一性、$A_0=0$、可预测性、递增性与递推构造。
- [University of Chicago, Probability Theory II, Theorem 2.6.7](https://math.uchicago.edu/~linus/6720sp20.pdf#page=21)：独立交叉核验同一分解及 $\Delta A_n=E(X_n\mid\mathcal F_{n-1})-X_{n-1}$ 的公式。

---
aliases:
  - "在列随机约定下有限 Markov 矩阵的稳态分布唯一当且仅当固定点空间维数为一"
  - Algebraic uniqueness criterion for stationary distributions
  - Markov 稳态唯一性判据
student_os: knowledge-atom
atom_id: LA-EIG-041
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov稳态分布]]"
  - "[[有限链平稳分解]]"
  - "[[Markov单位圆特征值半单]]"
related:
  - "[[有限不可约链稳态]]"
  - "[[稳态唯一不推收敛]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# 在列随机约定下有限 Markov 矩阵的稳态分布唯一当且仅当固定点空间维数为一
<!-- bilingual-en:start -->
*Under the column-stochastic convention, a finite Markov matrix has a unique stationary distribution exactly when its fixed-point space is one-dimensional*
<!-- bilingual-en:end -->

> [!summary] 有限 Markov 矩阵的代数判据
> 对有限列随机矩阵 $P$，稳态分布唯一当且仅当
> $$\dim\ker(P-I)=1.$$
> 行随机约定下，等价地检查左固定点空间 $\ker(P^T-I)$ 的维数。
> <!-- bilingual-en:start -->
> A finite column-stochastic matrix has a unique stationary distribution if and only if the eigenspace at eigenvalue one is one-dimensional.
> <!-- bilingual-en:end -->

[[Markov稳态分布]]是在固定点空间中再加非负与归一化。对有限 Markov 链，[[有限链平稳分解]]说明每个闭沟通类都提供一个极端稳态分布，全部稳态分布是这些分布的凸组合。因此固定点空间只有一个方向时，归一化后只能得到一个稳态分布；若固定点空间维数大于一，就会有多个稳态分布。

由[[Markov单位圆特征值半单]]可知，特征值 $1$ 对有限 Markov 矩阵总是半单的。因此这里的“固定点空间维数为一”也等价于“$1$ 是代数单根”。这个等价是 Markov 矩阵幂有界带来的，不能无条件套给任意方阵。

不可约是唯一性的常用充分条件，却不是必要条件。例如列随机矩阵
$$
P=\begin{bmatrix}1&1\\0&0\end{bmatrix}
$$
是可约的，但 $\ker(P-I)=\operatorname{span}\{(1,0)^T\}$，所以唯一稳态为 $(1,0)^T$。相反，当状态数至少为二时，$P=I$ 的固定点空间是整个空间，因而稳态不唯一。
<!-- bilingual-en:start -->
Finite-chain decomposition connects the eigenspace dimension with the number of closed classes. Irreducibility is sufficient for uniqueness but not necessary.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 一个有限列随机矩阵满足 $\dim\ker(P-I)=2$。它能否仍然只有一个稳态分布？
>
> **答案：** 不能。有限链的闭类分解会在这两个固定方向中产生多个非负归一化组合。

## 来源与核验

- [[有限链平稳分解]]：核对有限链闭类数、固定点空间与全部稳态分布的对应关系。
- [[Markov单位圆特征值半单]]：核对特征值 $1$ 的半单性及其有界幂依据。
- [[有限不可约链稳态]]：核对不可约是唯一性的充分条件而非必要条件。

---
aliases:
  - 在列随机约定下若有限 Markov 矩阵的幂收敛且稳态唯一则 $P^k$ 收敛到 $\pi\mathbf1^T$
  - Rank-one stationary limit of Markov matrix powers
  - Markov 幂的秩一极限
student_os: knowledge-atom
atom_id: LA-EIG-048
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov矩阵幂收敛判据]]"
  - "[[Markov稳态唯一判据]]"
  - "[[Markov稳态分布]]"
  - "[[Markov矩阵左右约定]]"
related:
  - "[[稳态唯一不推收敛]]"
  - "[[有限链逐步收敛]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# 在列随机约定下若有限 Markov 矩阵的幂收敛且稳态唯一则 $P^k$ 收敛到 $\pi\mathbf1^T$
<!-- bilingual-en:start -->
*For a finite column-stochastic matrix, convergent powers and a unique stationary distribution give the rank-one limit $\pi\mathbf1^T$*
<!-- bilingual-en:end -->

> [!summary] 核心定理
> 设 $P$ 是有限列随机矩阵，$P^k$ 收敛，并且归一化稳态分布 $\pi$ 唯一。则
> $$
> P^k\longrightarrow\pi\mathbf1^T,
> \qquad
> P^kp_0\longrightarrow\pi
> $$
> 对每个概率列向量 $p_0$ 成立。
>
> <!-- bilingual-en:start -->
> If a finite column-stochastic matrix has convergent powers and a unique normalized stationary distribution $\pi$, then $P^k\to\pi\mathbf1^T$, so every initial probability vector converges to $\pi$.
> <!-- bilingual-en:end -->

由[[Markov矩阵幂收敛判据]]，$P^k$ 的极限是到固定点空间 $E_1(P)$ 的谱投影。稳态唯一意味着 $E_1(P)$ 为一维，由归一化稳态向量 $\pi$ 张成。列随机约定给出配对左特征向量 $\mathbf1^T$，并且 $\mathbf1^T\pi=1$，所以这个谱投影正是
$$
\pi\mathbf1^T.
$$
对任意概率列向量 $p_0$，有 $\mathbf1^Tp_0=1$，于是
$$
(\pi\mathbf1^T)p_0=\pi.
$$

例如
$$
P=\begin{bmatrix}0.9&0.2\\0.1&0.8\end{bmatrix}
$$
的特征值为 $1$ 和 $0.7$，唯一稳态为 $\pi=(2/3,1/3)^T$，所以
$$
P^k\longrightarrow
\begin{bmatrix}2/3&2/3\\1/3&1/3\end{bmatrix}.
$$
只有稳态唯一不够；若仍存在 $-1$ 等其他单位根，矩阵幂会振荡，见[[稳态唯一不推收敛]]。
<!-- bilingual-en:start -->
Convergence leaves the spectral projection onto the fixed-point space. Uniqueness makes that space one-dimensional, and under the column convention the normalized right-left eigenvector pair is $\pi$ and $\mathbf1^T$, giving the projector $\pi\mathbf1^T$. Uniqueness alone is insufficient if another unit-circle eigenvalue remains.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 为什么 $(\pi\mathbf1^T)p_0=\pi$ 对每个概率列向量 $p_0$ 都成立？
>
> **答案：** 因为概率向量满足 $\mathbf1^Tp_0=1$。
>
> <!-- bilingual-en:start -->
> Why does $(\pi\mathbf1^T)p_0=\pi$ hold for every probability column vector $p_0$?
>
> **Answer:** Every probability vector satisfies $\mathbf1^Tp_0=1$.
> <!-- bilingual-en:end -->

## 来源与核验

- [Nick Higham, What Is a Stochastic Matrix?](https://nhigham.com/2022/12/13/what-is-a-stochastic-a-matrix/)：核对随机矩阵幂收敛的谱条件与稳态投影结构。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]]：核对列随机矩阵的稳态特征向量与次主特征值衰减。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S11_Lecture_Lecture_24_Markov_Matrices_Fourier_Series.pdf|MIT Lecture 24 transcript]]：核对列随机例中的模式分解与共同稳态极限。
<!-- bilingual-en:start -->
- [Nick Higham, What Is a Stochastic Matrix?](https://nhigham.com/2022/12/13/what-is-a-stochastic-a-matrix/) was checked for the spectral convergence condition for stochastic-matrix powers and the stationary projection structure.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]] was checked for the stationary eigenvector and decay of subdominant eigenmodes under the column convention.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S11_Lecture_Lecture_24_Markov_Matrices_Fourier_Series.pdf|MIT Lecture 24 transcript]] was checked for the mode decomposition and common stationary limit in the column-stochastic example.
<!-- bilingual-en:end -->

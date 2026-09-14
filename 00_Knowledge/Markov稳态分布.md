---
aliases:
  - "在列随机约定下满足固定点方程、非负且归一化的向量称为 Markov 稳态分布"
  - Stationary distribution as eigenvector
  - Markov 稳态分布
  - Markov 平稳分布
student_os: knowledge-atom
atom_id: LA-EIG-022
atom_set: eigenvalues-linear-dynamics
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov矩阵左右约定]]"
related:
  - "[[Markov稳态唯一判据]]"
  - "[[有限链平稳分解]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# 在列随机约定下满足固定点方程、非负且归一化的向量称为 Markov 稳态分布
<!-- bilingual-en:start -->
*Under the column-stochastic convention, a nonnegative normalized fixed point is called a Markov stationary distribution*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 采用列随机约定时，向量 $\pi$ 是 $P$ 的稳态分布，当且仅当
> $$P\pi=\pi,
> \qquad \pi_i\ge0,
> \qquad \mathbf1^T\pi=1.$$
> 行随机约定下，同一定义写成 $\pi^TP=\pi^T$、$\pi_i\ge0$、$\pi^T\mathbf1=1$。
> <!-- bilingual-en:start -->
> A stationary distribution is a fixed vector that is nonnegative and normalized to have total mass one. Its side depends on the stochastic convention.
> <!-- bilingual-en:end -->

固定点方程表达“再走一步，分布不变”；非负和归一化则保证它真的是概率分布。三项缺一不可。只解 $(P-I)x=0$ 得到的是一个线性子空间：非零倍数仍是特征向量，其中可能有负分量，也可能总和不等于一。

例如
$$
P=\begin{bmatrix}0.9&0.2\\0.1&0.8\end{bmatrix}
$$
是列随机矩阵。解 $(P-I)x=0$ 得到 $x\propto(2,1)^T$；归一化后才得到
$$
\pi=(2/3,1/3)^T.
$$
向量 $(2,1)^T$ 已经满足固定点方程，却还不是概率分布；它的负倍数也满足同一线性方程，更不能当作分布。
<!-- bilingual-en:start -->
Solving the eigenspace gives a direction, not automatically a probability distribution. The vector $(2,1)^T$ must be normalized to $(2/3,1/3)^T$, and a negative multiple is algebraically valid but probabilistically invalid.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 已求得特征值 $1$ 的向量 $x=(3,6,0)^T$。相应的稳态候选是什么？
>
> **答案：** $(1/3,2/3,0)^T$；还要确认所用左右约定和固定点方程相匹配。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]]：核对稳态为特征值 $1$ 的特征向量及两状态归一化。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S11_Recitation_Problem_Solving_Markov_Matrices.pdf|MIT Session 2.11 recitation]]：核对概率向量随矩阵幂演化及稳态计算。

---
aliases:
  - 有限 Markov 矩阵必有特征值一
  - Every finite stochastic matrix has eigenvalue one
  - Markov 矩阵的特征值一
student_os: knowledge-atom
atom_id: LA-EIG-046
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
requires:
  - "[[Markov矩阵]]"
  - "[[特征对]]"
related:
  - "[[Markov矩阵左右约定]]"
  - "[[Markov矩阵谱边界]]"
  - "[[Markov稳态分布]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

<!-- bilingual-en:start -->
# Why a Markov matrix has eigenvalue one
<!-- bilingual-en:end -->

*为什么 Markov 矩阵必有特征值一*

<!-- bilingual-en:start -->
A Markov transition preserves total probability. This conservation law forces the number $1$ to be an eigenvalue of every finite Markov matrix. With one convention the corresponding vector is immediately visible on the right; with the other it first appears on the left, so we need one more linear-algebra step.
<!-- bilingual-en:end -->

Markov 转移保持总概率不变。这个守恒关系保证每个有限 Markov 矩阵都有特征值 $1$。一种约定能直接看见右边的对应向量；另一种先在左边看见它，需要再补一步线性代数论证。

<!-- bilingual-en:start -->
Recall that an [[特征对|eigenvalue]] $\lambda$ means there is a **nonzero** vector $v$ with $Pv=\lambda v$. When $\lambda=1$, that vector is unchanged by $P$. For the column-stochastic example from [[Markov矩阵|Markov matrices]],
<!-- bilingual-en:end -->

回忆[[特征对|特征值]] $\lambda$ 的含义：存在**非零**向量 $v$ 使 $Pv=\lambda v$。$\lambda=1$ 就表示这个向量经过 $P$ 后保持不变。对[[Markov矩阵]]中的列随机例子，

$$
P=\begin{bmatrix}0.9&0.2\\0.1&0.8\end{bmatrix},\qquad
P\begin{bmatrix}2\\1\end{bmatrix}
=\begin{bmatrix}2\\1\end{bmatrix}.
$$

<!-- bilingual-en:start -->
So $1$ is an eigenvalue here. Notice that the unchanged column is $(2,1)^T$, not the all-ones column $(1,1)^T$. The superscript $T$ denotes [[矩阵转置|transpose]]. The following proof explains why some nonzero unchanged column always exists, without first solving for it.
<!-- bilingual-en:end -->

所以这里确实有特征值 $1$。注意不变的列是 $(2,1)^T$，不是全一列 $(1,1)^T$。上标 $T$ 表示[[矩阵转置]]。下面的证明说明为什么总能找到某个非零不变列，而不必先把它解出来。

> [!important] Finite stochastic matrices · 有限随机矩阵
> <!-- bilingual-en:start -->
> Let $P$ be a real $n\times n$ matrix with $n\ge1$, nonnegative entries, and either all row sums or all column sums equal to $1$. Then $1$ is an eigenvalue of $P$. No strict positivity of every entry is required.
> <!-- bilingual-en:end -->
>
> 设 $P$ 是实 $n\times n$ 矩阵，$n\ge1$，各元素非负，而且每行之和为一或每列之和为一。那么 $1$ 是 $P$ 的特征值。不要求每个元素都严格为正。

<!-- bilingual-en:start -->
## Row sums give a right eigenvector
<!-- bilingual-en:end -->

*行和直接给出右特征向量*

<!-- bilingual-en:start -->
Let $\mathbf1=(1,\ldots,1)^T$, which is nonzero because $n\ge1$. If $P$ is row-stochastic, the $i$-th component of $P\mathbf1$ is the sum of row $i$:
<!-- bilingual-en:end -->

令 $\mathbf1=(1,\ldots,1)^T$；由于 $n\ge1$，它不是零向量。若 $P$ 是行随机矩阵，$P\mathbf1$ 的第 $i$ 个分量就是第 $i$ 行之和：

$$
(P\mathbf1)_i=\sum_{j=1}^{n}p_{ij}=1,
\qquad\text{hence}\qquad P\mathbf1=\mathbf1.
$$

<!-- bilingual-en:start -->
This is already the eigenvector equation with $v=\mathbf1$ and $\lambda=1$.
<!-- bilingual-en:end -->

这已经是 $v=\mathbf1$、$\lambda=1$ 的特征向量方程。

<!-- bilingual-en:start -->
## Column sums first give a left eigenvector
<!-- bilingual-en:end -->

*列和先给出左特征向量*

<!-- bilingual-en:start -->
If $P$ is column-stochastic, adding each column gives $\mathbf1^TP=\mathbf1^T$. Transposing this equation gives $P^T\mathbf1=\mathbf1$, so $1$ is an eigenvalue of $P^T$. This is the [[Markov矩阵左右约定|left–right convention]] distinction; we cannot simply replace $P^T$ by $P$ in the vector equation.
<!-- bilingual-en:end -->

若 $P$ 是列随机矩阵，对每列相加得到 $\mathbf1^TP=\mathbf1^T$。把等式转置，得到 $P^T\mathbf1=\mathbf1$，所以先知道 $1$ 是 $P^T$ 的特征值。这正是[[Markov矩阵左右约定|左右约定]]的区别，不能直接把向量方程中的 $P^T$ 换成 $P$。

<!-- bilingual-en:start -->
To transfer the eigenvalue, use the [[特征多项式|characteristic polynomial]] $\chi_P(t)=\det(tI-P)$, whose roots are the eigenvalues. Here $t$ is a scalar variable, $I$ is the [[单位矩阵|identity matrix]], and $\det$ denotes the determinant. Since [[行列式转置不变性|transposing a square matrix leaves its determinant unchanged]],
<!-- bilingual-en:end -->

为了把特征值传回去，使用[[特征多项式]] $\chi_P(t)=\det(tI-P)$：它的根就是特征值。这里 $t$ 是标量变量，$I$ 是[[单位矩阵]]，$\det$ 表示行列式。由[[行列式转置不变性|方阵转置不改变行列式]]，

$$
\chi_{P^T}(t)=\det(tI-P^T)
=\det\bigl((tI-P)^T\bigr)
=\det(tI-P)=\chi_P(t).
$$

<!-- bilingual-en:start -->
Thus $P$ and $P^T$ have the same characteristic polynomial. Since $\chi_{P^T}(1)=0$, also $\chi_P(1)=0$, which supplies a nonzero vector $v$ with $Pv=v$. The probability entries are real, and such a vector can be chosen real by solving the real homogeneous system $(P-I)v=0$.
<!-- bilingual-en:end -->

因此 $P$ 与 $P^T$ 的特征多项式相同。既然 $\chi_{P^T}(1)=0$，就有 $\chi_P(1)=0$，于是存在非零向量 $v$ 使 $Pv=v$。概率矩阵的元素为实数，解实齐次方程组 $(P-I)v=0$ 就可以选出实向量。

<!-- bilingual-en:start -->
## What the proof does and does not determine
<!-- bilingual-en:end -->

*从特征值存在到后续问题*

<!-- bilingual-en:start -->
For the opening example, $(P-I)v=0$ reduces to $-0.1v_1+0.2v_2=0$, so $v_1=2v_2$. Choosing $v=(2,1)^T$ and dividing by its component sum gives $\pi=(2/3,1/3)^T$. A [[Markov稳态分布|stationary distribution]] must satisfy both the fixed-point equation and the probability conditions: nonnegative components and sum $1$. An arbitrary eigenvector supplied by a linear solver need not already satisfy those conditions; even $(-2,-1)^T$ is an eigenvector in this example.
<!-- bilingual-en:end -->

开篇例子的 $(P-I)v=0$ 化为 $-0.1v_1+0.2v_2=0$，即 $v_1=2v_2$。选 $v=(2,1)^T$ 并除以分量之和，得到 $\pi=(2/3,1/3)^T$。[[Markov稳态分布|稳态分布]]除了满足固定点方程，还必须满足概率条件：分量非负、总和为一。线性求解器给出的任意特征向量未必已经满足这些条件；上例的 $(-2,-1)^T$ 也同样是特征向量。

> [!warning] Existence is separate from uniqueness and convergence · 存在不等于唯一或收敛
> <!-- bilingual-en:start -->
> For $I_2$, every vector is fixed, so the fixed-point space is not one-dimensional. For $S=\begin{bmatrix}0&1\\1&0\end{bmatrix}$, eigenvalue $1$ exists, yet successive updates of $(1,0)^T$ alternate between $(1,0)^T$ and $(0,1)^T$. The theorem therefore does not by itself give a unique stationary distribution or convergence of every initial distribution.
> <!-- bilingual-en:end -->
>
> 对 $I_2$，每个向量都保持不变，固定点空间并不是一维。对 $S=\begin{bmatrix}0&1\\1&0\end{bmatrix}$，虽然有特征值 $1$，但从 $(1,0)^T$ 出发会在 $(1,0)^T$ 与 $(0,1)^T$ 之间交替。因此，本定理本身没有给出稳态唯一性，也没有保证每种初始分布都会收敛。

<!-- bilingual-en:start -->
The location of the other eigenvalues is treated in the [[Markov矩阵谱边界|spectral bound]], while [[Markov稳态唯一判据|stationary-distribution uniqueness]] and [[稳态唯一不推收敛|the distinction between uniqueness and convergence]] address different questions. The proof above only uses the unit-sum identities; nonnegativity is additionally what makes the matrix a valid probability transition.
<!-- bilingual-en:end -->

其他特征值的位置见 [[Markov矩阵谱边界]]；[[Markov稳态唯一判据|稳态唯一性]]与[[稳态唯一不推收敛|唯一和收敛的区别]]回答的是不同问题。上面的代数证明只用了和为一的等式；非负性则进一步保证矩阵真能表示概率转移。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]], p.1, and [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S11_Lecture_Lecture_24_Markov_Matrices_Fourier_Series.pdf|Lecture 24 transcript]], PDF pp.4–7: the column-sum argument and equality of the characteristic polynomials under transposition. The proof above explicitly separates left and right vectors.
- [Levin and Peres, with Wilmer, *Markov Chains and Mixing Times*, 2nd ed.](https://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf#page=18), §1.1 printed pp.2–5 / PDF pp.18–21 and §1.5.1 printed p.9 / PDF p.25: row-stochastic transitions and the probability requirement for stationary distributions. The identity, swapping, and two-state products are checked directly.
- [Higham, *What Is a Stochastic Matrix?*](https://nhigham.com/2022/12/13/what-is-a-stochastic-a-matrix/): supplementary discussion of the all-ones eigenvector under the row convention.
<!-- bilingual-en:end -->

- MIT 18.06SC Session 2.11 summary，第1页，以及 Lecture 24讲稿PDF第4–7页：支持列和论证及转置前后特征多项式相同；本文明确分开左右向量。
- Levin、Peres 与 Wilmer，《Markov Chains and Mixing Times》第2版 §1.1，印刷第2–5页 / PDF第18–21页；§1.5.1，印刷第9页 / PDF第25页：支持行随机转移与稳态分布的概率条件。单位矩阵、交换矩阵及两状态乘积直接核算。
- Higham《What Is a Stochastic Matrix?》：补充行随机约定中全一特征向量的说明。

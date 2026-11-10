---
aliases:
- Stationary distribution
- Invariant distribution of a Markov chain
- Markov 平稳分布
- Markov 稳态分布
student_os: knowledge-atom
atom_id: LA-EIG-022
atom_set: eigenvalues-linear-dynamics
atom_type: definition
status: source-checked
requires:
- '[[Markov矩阵左右约定]]'
related:
- '[[Markov稳态唯一判据]]'
- '[[有限链平稳分解]]'
part_of:
- '[[特征值、对角化与线性动力系统.canvas]]'
---

<!-- bilingual-en:start -->
# Stationary distributions
<!-- bilingual-en:end -->

*Markov 稳态分布*

<!-- bilingual-en:start -->
A **stationary distribution** is a probability allocation that remains unchanged after one transition of a Markov chain. The individual state can still change: stationarity says that the proportions across possible states stay the same. It defines a fixed distribution, without claiming that every starting distribution eventually reaches it.
<!-- bilingual-en:end -->

**稳态分布**（也叫平稳分布）是一种经过一步 Markov 转移后保持不变的概率分配。具体状态仍然可以变化；不变的是各个可能状态所占的概率比例。定义说的是一个固定分布，没有声称任意起始分布最终都会趋近它。

> [!example] Balanced movement between two states · 两个状态之间的流入流出平衡
> <!-- bilingual-en:start -->
> Use the [[Markov矩阵左右约定|column-stochastic convention]], so entry $P_{ij}$ is the probability of moving from state $j$ to state $i$. Let
>
> $$P=\begin{pmatrix}0.9&0.2\\0.1&0.8\end{pmatrix},\qquad
> \pi=\begin{pmatrix}2/3\\1/3\end{pmatrix}.$$
>
> Of the total probability, $(2/3)\cdot0.1=1/15$ moves from state one to state two, and $(1/3)\cdot0.2=1/15$ moves back. Hence
>
> $$P\pi=\begin{pmatrix}0.9(2/3)+0.2(1/3)\\0.1(2/3)+0.8(1/3)\end{pmatrix}
> =\begin{pmatrix}2/3\\1/3\end{pmatrix}=\pi.$$
>
> These proportions stay fixed even though there is movement. They are not uniform: the two states need not have equal probabilities.
> <!-- bilingual-en:end -->
>
> 采用[[Markov矩阵左右约定|列随机约定]]，条目 $P_{ij}$ 表示从状态 $j$ 移到状态 $i$ 的概率。对上面的 $P$ 和 $\pi$，总概率中有 $(2/3)\cdot0.1=1/15$ 从状态一流向状态二，同时有 $(1/3)\cdot0.2=1/15$ 流回。因此直接相乘得到 $P\pi=\pi$。存在实际移动，但比例保持不变；这个分布并不均匀，两状态概率不必相等。

<!-- bilingual-en:start -->
## Three requirements, not just an eigenvector equation
<!-- bilingual-en:end -->

*三个条件，而不只是特征向量方程*

<!-- bilingual-en:start -->
Let $P$ be a fixed [[Markov矩阵|column-stochastic matrix]] of size $n\times n$, with integer $n\ge1$. A probability column lists the probabilities of the $n$ states, as in a finite [[概率质量函数|probability mass function]]. A column $\pi=(\pi_1,\ldots,\pi_n)^T$ is stationary precisely when

$$P\pi=\pi,\qquad \pi_i\ge0\ (1\le i\le n),\qquad
\mathbf1^T\pi=\sum_{i=1}^n\pi_i=1.$$

Here $T$ denotes [[矩阵转置|transpose]] and $\mathbf1$ is the column of $n$ ones. The first condition is the fixed-point equation; the others make the fixed point a valid probability distribution. If $p_0=\pi$, then $p_k=P^k\pi=\pi$ for every integer $k\ge0$.
<!-- bilingual-en:end -->

设 $P$ 为固定的 $n\times n$ [[Markov矩阵|列随机矩阵]]，整数 $n\ge1$。概率列把 $n$ 个状态的概率依次列出，就像有限[[概率质量函数]]。列 $\pi=(\pi_1,\ldots,\pi_n)^T$ 恰好在同时满足上面三个条件时平稳。这里 $T$ 表示[[矩阵转置|转置]]，$\mathbf1$ 是由 $n$ 个一组成的列。第一条是固定点方程，后两条保证它确实是概率分布。若从 $p_0=\pi$ 开始，则全部整数 $k\ge0$ 下都有 $p_k=P^k\pi=\pi$。

<!-- bilingual-en:start -->
Under the row convention, write the transition matrix as $R=P^T$ and the probability row as $\pi^T$. The same statement becomes $\pi^TR=\pi^T$, together with nonnegative entries and $\pi^T\mathbf1=1$. A row-stochastic matrix denoted by $P$ would instead use $\pi^TP=\pi^T$. The multiplication side follows the convention, not the letter chosen for the matrix.
<!-- bilingual-en:end -->

采用行约定时，把转移矩阵记为 $R=P^T$，概率行记为 $\pi^T$，同一定义就写成 $\pi^TR=\pi^T$，并要求各元素非负、$\pi^T\mathbf1=1$。若行随机矩阵也记作 $P$，则写 $\pi^TP=\pi^T$。乘法在哪一侧，由约定决定，不由矩阵使用哪个字母决定。

<!-- bilingual-en:start -->
## Solve the fixed-point equation, then check probability
<!-- bilingual-en:end -->

*先解固定点，再核对概率条件*

<!-- bilingual-en:start -->
For the opening matrix, $(P-I_2)x=0$ reduces to $-0.1x_1+0.2x_2=0$, hence $x_1=2x_2$. Here $I_2$ is the [[单位矩阵|identity matrix]]. The [[零空间|null space]] of $P-I_2$ is the line of multiples of $(2,1)^T$. The vector $(2,1)^T$ is already an [[特征对|eigenvector]] for eigenvalue one but has total mass three. Dividing by three gives the displayed stationary distribution. Its negative multiple satisfies the same linear equation but cannot represent probabilities.
<!-- bilingual-en:end -->

对开头的矩阵，$(P-I_2)x=0$ 化成 $-0.1x_1+0.2x_2=0$，即 $x_1=2x_2$，其中 $I_2$ 是[[单位矩阵]]。所以 $P-I_2$ 的[[零空间]]是 $(2,1)^T$ 的倍数构成的直线。$(2,1)^T$ 已是特征值一的[[特征对|特征向量]]，但总质量为三；除以三才得到展示的稳态分布。它的负倍数也满足同一个线性方程，却不能表示概率。

<!-- bilingual-en:start -->
With more fixed directions, an arbitrary eigenvector can contain both positive and negative entries. Dividing by its coordinate sum does not remove mixed signs, and a zero sum cannot be used as a divisor. Solve the fixed-point equations together with nonnegativity and total mass one. If a nonnegative fixed vector is $x=(3,6,0)^T$, for example, its sum is nine and normalization gives $(1/3,2/3,0)^T$. A stationary distribution is allowed to assign zero probability to some states.
<!-- bilingual-en:end -->

若固定方向不止一个，随意取出的特征向量还可能同时有正负元素；除以坐标和不能消除这种混合符号，坐标和为零时更不能拿来作除数。因此应把固定点方程、非负性与总和一共同求解。例如，已确认非负固定向量 $x=(3,6,0)^T$ 后，其和为九，归一化得到 $(1/3,2/3,0)^T$。稳态分布允许某些状态的概率为零。

<!-- bilingual-en:start -->
Every finite Markov chain has [[有限Markov链稳态存在|at least one stationary distribution]], even if some states cannot reach each other or transitions cycle. The linked theorem explains why averaging over time produces an exact stationary limit along a subsequence. Existence alone does not answer uniqueness or convergence from other starting distributions.
<!-- bilingual-en:end -->

每个有限 Markov 链都[[有限Markov链稳态存在|至少有一个稳态分布]]，即使部分状态互相到不了或转移存在循环。链接中的定理解释了为什么时间平均可以沿子序列产生精确的稳态极限。单靠存在性，还不能回答是否唯一或从其他起点是否收敛。

> [!warning] Existence, uniqueness and convergence are different · 存在、唯一、收敛是三个问题
> <!-- bilingual-en:start -->
> For $P=I_2$, every probability column $(a,1-a)^T$, $0\le a\le1$, is stationary, so uniqueness fails. For the swap matrix $F=\begin{pmatrix}0&1\\1&0\end{pmatrix}$, the unique stationary column is $(1/2,1/2)^T$, yet the starting column $(1,0)^T$ alternates forever. Starting from the stationary column itself does stay stationary. Use [[Markov稳态唯一判据|the uniqueness criterion]] and [[Markov矩阵幂收敛判据|the convergence criterion]] for these further questions; [[有限链平稳分解|closed-class decomposition]] describes all stationary distributions when there are several.
> <!-- bilingual-en:end -->
>
> 对 $P=I_2$，每个概率列 $(a,1-a)^T$、$0\le a\le1$ 都平稳，所以不唯一。对交换矩阵 $F$，唯一稳态列为 $(1/2,1/2)^T$，但从 $(1,0)^T$ 开始却永远交替；从稳态列本身出发，当然会一直保持稳态。进一步的问题分别由[[Markov稳态唯一判据]]和[[Markov矩阵幂收敛判据]]处理；存在多个稳态时，[[有限链平稳分解|闭类分解]]描述其全部形式。

## 来源与核验

<!-- bilingual-en:start -->
- [Levin–Peres with Wilmer, Markov Chains and Mixing Times, 2nd ed., §1.5.1, PDF pp.25–26 / printed pp.9–10; Proposition 1.32, PDF p.36 / printed p.20](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf#page=25): probability fixed-point definition and finite-state existence, including reducible chains. The book uses row probabilities; this card writes the equivalent column convention. The complete existence argument is in the linked theorem.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]], pp.2–3: the two-state matrix, eigenvector $(2,1)^T$ and its normalized proportions. [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S11_Recitation_Problem_Solving_Markov_Matrices.pdf|The recitation]], pp.1–3, supplies a separate two-state probability calculation. The balance, normalization, identity and swap examples here are checked directly.
<!-- bilingual-en:end -->

- Levin–Peres 等第2版，§1.5.1，PDF第25–26页／印刷9–10页；命题1.32，PDF第36页／印刷20页：核对概率固定点定义及包括可约链在内的有限状态存在性。原书使用概率行，本文采用等价的概率列约定；完整存在性证明由链接的定理承担。
- MIT第2.11节摘要第2–3页：核对两状态矩阵、$(2,1)^T$ 特征向量及归一化比例。习题讲解第1–3页提供另一组两状态概率演算。本文流量平衡、归一化、单位矩阵和交换矩阵例子均直接核算。

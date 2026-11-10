---
aliases:
  - Row-stochastic and column-stochastic conventions
  - Markov 矩阵左右特征向量约定
student_os: knowledge-atom
atom_id: LA-EIG-021
atom_set: eigenvalues-linear-dynamics
atom_type: distinction
status: source-checked
requires:
  - "[[Markov矩阵]]"
  - "[[特征对]]"
related:
  - "[[Markov稳态分布]]"
  - "[[Markov矩阵必有特征值一]]"
  - "[[Markov矩阵谱边界]]"
  - "[[DTMC时间齐次转移核]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

<!-- bilingual-en:start -->
# Row and column conventions for Markov matrices
<!-- bilingual-en:end -->

*Markov 矩阵的行列约定与左右位置*

<!-- bilingual-en:start -->
The same transition rule can be written with probability **columns** or probability **rows**. Both are valid. The choice changes where the current state is indexed, which side the distribution multiplies, and where the all-ones and stationary vectors appear. A consistent convention keeps all three choices together.
<!-- bilingual-en:end -->

同一套转移规则既可以用概率**列向量**写，也可以用概率**行向量**写。两种都成立，但当前状态放在哪个下标、分布从哪一侧相乘，以及全一向量和稳态向量出现在哪边，都要一起配套。

<!-- bilingual-en:start -->
## Transpose the whole description
<!-- bilingual-en:end -->

*把整套描述一起转置*

<!-- bilingual-en:start -->
A [[Markov矩阵|Markov matrix]] has nonnegative entries and unit sums in its specified direction. Suppose the two states are $A,B$: from $A$ the probabilities of the next state are $(0.9,0.1)$; from $B$ they are $(0.2,0.8)$. The following matrices describe exactly these same transitions:
<!-- bilingual-en:end -->

[[Markov矩阵]]的元素非负，并且指定方向上的和为一。设状态为 $A,B$：从 $A$ 出发，下一步的概率是 $(0.9,0.1)$；从 $B$ 出发，则是 $(0.2,0.8)$。下面两个矩阵描述完全相同的转移：

$$
P=\begin{bmatrix}0.9&0.2\\0.1&0.8\end{bmatrix},\qquad
R=P^T=\begin{bmatrix}0.9&0.1\\0.2&0.8\end{bmatrix}.
$$

<!-- bilingual-en:start -->
In the column-stochastic matrix $P$, a **column** fixes the starting state; in the row-stochastic matrix $R$, a **row** fixes it. The superscript $T$ denotes [[矩阵转置|transpose]], exchanging rows and columns. For the current probability column $x=(0.6,0.4)^T$, the corresponding probability row is $p=x^T=(0.6,0.4)$. Each has nonnegative components with sum $1$. Their updates are
<!-- bilingual-en:end -->

列随机矩阵 $P$ 用一**列**固定出发状态；行随机矩阵 $R$ 用一**行**固定出发状态。上标 $T$ 表示[[矩阵转置]]，交换行与列。当前概率列为 $x=(0.6,0.4)^T$ 时，对应概率行为 $p=x^T=(0.6,0.4)$。两者都要求分量非负且总和为 $1$。更新分别是

$$
x'=Px=\begin{bmatrix}0.62\\0.38\end{bmatrix},\qquad
p'=pR=\begin{bmatrix}0.62&0.38\end{bmatrix}=(x')^T.
$$

<!-- bilingual-en:start -->
The primes mean ‘next step’. The identity $(Px)^T=x^TP^T$ explains why the results agree: transposing a [[乘积的转置|product reverses its order]]. Merely turning $x$ into a row while leaving $P$ unchanged would instead give $x^TP=(0.58,0.44)$, whose sum is $1.02$; that mixes the conventions.
<!-- bilingual-en:end -->

撇号表示“下一步”。由[[乘积的转置|乘积转置会反转次序]]，$(Px)^T=x^TP^T$，所以两种结果一致。如果只把 $x$ 横过来而保持 $P$ 不变，反而得到 $x^TP=(0.58,0.44)$，总和为 $1.02$；这就是混用了约定。

<!-- bilingual-en:start -->
## Reading the indices
<!-- bilingual-en:end -->

*看清下标中的出发与到达*

<!-- bilingual-en:start -->
For a positive integer $n$ states, let $X_k$ denote the state at step $k$. We use $P$ for the column convention and $R=P^T$ for the row convention throughout this card. Their entries specify the transitions as follows:
<!-- bilingual-en:end -->

对正整数 $n$ 个状态，用 $X_k$ 表示第 $k$ 步的状态。本卡始终用 $P$ 表示列随机约定，用 $R=P^T$ 表示行随机约定。元素指定的转移为

$$
\begin{aligned}
P_{ij}&=\Pr(X_{k+1}=i\mid X_k=j),& x_{k+1}&=Px_k,\\
R_{ij}&=\Pr(X_{k+1}=j\mid X_k=i),& p_{k+1}&=p_kR.
\end{aligned}
$$

<!-- bilingual-en:start -->
The conditional notation means “next state, given the current state”: $P_{ij}$ goes from $j$ to $i$, whereas $R_{ij}$ goes from $i$ to $j$. The transition rule is specified for every possible starting state, even if that state currently has probability zero; ordinary event-conditioning identifies it when the conditioning event has positive probability. Reusing the same matrices for every $k$ is the [[DTMC时间齐次转移核|time-homogeneous model]].
<!-- bilingual-en:end -->

条件概率记号表示“给定当前状态后，下一步到某状态的概率”：$P_{ij}$ 是从 $j$ 到 $i$，而 $R_{ij}$ 是从 $i$ 到 $j$。转移规则会对每个可能出发状态都作出规定，即使该状态目前的概率为零；当条件事件具有正概率时，通常的事件条件概率就与该规则对应。对每个 $k$ 都重复使用同一矩阵，是[[DTMC时间齐次转移核|时间齐次模型]]。

<!-- bilingual-en:start -->
## Conserved total probability and stationary distributions
<!-- bilingual-en:end -->

*守恒的总概率与不变的概率分布*

<!-- bilingual-en:start -->
Let $\mathbf1=(1,\ldots,1)^T$ be the all-ones column. Multiplying $\mathbf1^Tx$ adds the components of a column, while $p\mathbf1$ adds those of a row. Unit column sums and unit row sums therefore give
<!-- bilingual-en:end -->

令 $\mathbf1=(1,\ldots,1)^T$ 为全一列向量。$\mathbf1^Tx$ 把列向量的分量相加，$p\mathbf1$ 把行向量的分量相加。因此，列和为一与行和为一分别给出

$$
\mathbf1^TP=\mathbf1^T,\qquad R\mathbf1=\mathbf1.
$$

<!-- bilingual-en:start -->
These identities prove preservation of total probability: $\mathbf1^Tx_{k+1}=\mathbf1^TPx_k=\mathbf1^Tx_k=1$, or $p_{k+1}\mathbf1=p_kR\mathbf1=p_k\mathbf1=1$. Nonnegativity of the entries also preserves nonnegativity of the distribution.
<!-- bilingual-en:end -->

这些等式证明总概率保持不变：$\mathbf1^Tx_{k+1}=\mathbf1^TPx_k=\mathbf1^Tx_k=1$，或 $p_{k+1}\mathbf1=p_kR\mathbf1=p_k\mathbf1=1$。矩阵元素非负还保证分布的非负性被保留。

<!-- bilingual-en:start -->
A [[Markov稳态分布|stationary distribution]] is a probability distribution unchanged after one step. Writing it as a column $\pi$ in the first convention or a row $\pi^T$ in the second gives
<!-- bilingual-en:end -->

[[Markov稳态分布|稳态分布]]是再走一步仍然不变的概率分布。第一种约定下写成列 $\pi$，第二种写成行 $\pi^T$，分别得到

$$
P\pi=\pi,\qquad \pi^TR=\pi^T,\qquad
\pi_i\ge0,\qquad \sum_{i=1}^{n}\pi_i=1.
$$

<!-- bilingual-en:start -->
For a square matrix $M$ and scalar $\lambda$, a nonzero column $v$ satisfying $Mv=\lambda v$ is a right [[特征对|eigenvector]], and a nonzero row $w^T$ satisfying $w^TM=\lambda w^T$ is a left eigenvector. Thus, for $P$, the all-ones vector appears on the **left** and a stationary distribution on the **right**; for $R$ these positions reverse. In the two-state example, $\pi=(2/3,1/3)^T$ satisfies the displayed stationary equations.
<!-- bilingual-en:end -->

对方阵 $M$ 和标量 $\lambda$，非零列 $v$ 满足 $Mv=\lambda v$ 时称为右[[特征对|特征向量]]；非零行 $w^T$ 满足 $w^TM=\lambda w^T$ 时称为左特征向量。因此，对 $P$，全一向量出现在**左边**，稳态分布出现在**右边**；对 $R$ 则反过来。两状态例子中的 $\pi=(2/3,1/3)^T$ 满足上述稳态方程。

> [!important] The two vectors do different jobs · 两种向量承担不同作用
> <!-- bilingual-en:start -->
> $\mathbf1$ adds up probabilities; it is not itself a probability distribution unless there is only one state. Even normalizing it need not give a stationary distribution. For the example, $P(1/2,1/2)^T=(0.55,0.45)^T$, so the uniform probabilities change. The fixed-point vector and the vector used to compute total mass should not be interchanged.
> <!-- bilingual-en:end -->
>
> $\mathbf1$ 用来把概率加起来；除非只有一个状态，它自身并不是概率分布。即使把它归一化，也未必得到稳态分布。上例有 $P(1/2,1/2)^T=(0.55,0.45)^T$，因此均匀概率会发生变化。固定点向量与负责计算总质量的向量不能互换。

<!-- bilingual-en:start -->
This left–right distinction does not change the fact that [[Markov矩阵必有特征值一|every finite Markov matrix has eigenvalue one]]. It changes which eigenvector is visible directly from the unit-sum condition.
<!-- bilingual-en:end -->

左右位置的区别不影响[[Markov矩阵必有特征值一|有限 Markov 矩阵必有特征值一]]这一事实，只改变从和为一的条件能直接看见哪一侧的特征向量。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Levin and Peres, with Wilmer, *Markov Chains and Mixing Times*, 2nd ed.](https://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf#page=18), §1.1 printed pp.2–5 / PDF pp.18–21 and §1.5.1 printed p.9 / PDF p.25: current/next-state indices, row evolution, and stationary distributions.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]], pp.1–3, and [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S11_Lecture_Lecture_24_Markov_Matrices_Fourier_Series.pdf|Lecture 24 transcript]], PDF pp.1–7: the column convention, conserved total, and left/right roles.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S11_Recitation_Problem_Solving_Markov_Matrices.pdf|MIT Session 2.11 recitation]], PDF pp.1–3: probability-column iteration and normalization in a two-state example. The numerical products in this card are checked directly.
- [Higham, *What Is a Stochastic Matrix?*](https://nhigham.com/2022/12/13/what-is-a-stochastic-a-matrix/): supplementary comparison of row and column conventions.
<!-- bilingual-en:end -->

- Levin、Peres 与 Wilmer，《Markov Chains and Mixing Times》第2版 §1.1，印刷第2–5页 / PDF第18–21页；§1.5.1，印刷第9页 / PDF第25页：支持当前与下一状态下标、概率行更新及稳态分布。
- MIT 18.06SC Session 2.11 summary，第1–3页，以及 Lecture 24讲稿PDF第1–7页：支持列随机约定、总量守恒和左右位置。
- MIT Session 2.11习题课讲稿PDF第1–3页：支持概率列的迭代与两状态例子的归一化；本文数值乘积均直接复算。
- Higham《What Is a Stochastic Matrix?》：补充行列约定的比较。

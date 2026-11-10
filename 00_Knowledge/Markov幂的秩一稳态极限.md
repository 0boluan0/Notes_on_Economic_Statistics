---
aliases:
  - Rank-one stationary limit of Markov matrix powers
  - Markov 幂的秩一极限
student_os: knowledge-atom
atom_id: LA-EIG-048
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
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

<!-- bilingual-en:start -->
# A common stationary limit has rank one
<!-- bilingual-en:end -->

*Markov 矩阵幂的秩一稳态极限*

<!-- bilingual-en:start -->
When a finite Markov chain approaches the **same probability distribution from every starting state**, its limiting transition matrix repeats that distribution in every starting-state column. This repeated-column form is the meaning of the rank-one stationary limit. Two ingredients are needed: the powers must converge, and the stationary distribution must be unique.
<!-- bilingual-en:end -->

当有限 Markov 链从**每个出发状态都趋向同一个概率分布**时，它的极限转移矩阵会在每个出发状态对应的列里重复这个分布。这种重复列的形式，就是秩一稳态极限的含义。需要两件事同时成立：矩阵幂收敛，以及稳态分布唯一。

<!-- bilingual-en:start -->
## Two different starts approach the same proportions
<!-- bilingual-en:end -->

*不同出发状态，趋向同一组比例*

<!-- bilingual-en:start -->
Take the [[Markov矩阵|column-stochastic matrix]] below: starting in state 1 gives next-step probabilities $(0.9,0.1)^T$; starting in state 2 gives $(0.2,0.8)^T$. Here $T$ denotes [[矩阵转置|transpose]], turning a coordinate row into a column.
<!-- bilingual-en:end -->

取下面的[[Markov矩阵|列随机矩阵]]：从状态1出发，下一步概率为 $(0.9,0.1)^T$；从状态2出发，则为 $(0.2,0.8)^T$。这里 $T$ 表示[[矩阵转置]]，把横写的坐标变成列。

$$
P=\begin{pmatrix}0.9&0.2\\0.1&0.8\end{pmatrix},\qquad
\pi=\begin{pmatrix}2/3\\1/3\end{pmatrix},\qquad
L=\begin{pmatrix}2/3&2/3\\1/3&1/3\end{pmatrix}.
$$

<!-- bilingual-en:start -->
A [[Markov稳态分布|stationary distribution]] is a nonnegative probability column whose entries sum to one and which satisfies $P\pi=\pi$. Here the first fixed-point equation is $0.9\pi_1+0.2\pi_2=\pi_1$, giving $\pi_1=2\pi_2$; normalization fixes the displayed $\pi$ uniquely. The eigenvalues of $P$ are $1$ and $0.7$, so the [[Markov矩阵幂收敛判据|matrix-power convergence criterion]] applies. The limit is $L$: both starting states eventually give probabilities $2/3$ and $1/3$.
<!-- bilingual-en:end -->

[[Markov稳态分布]]是非负、分量和为一，并满足 $P\pi=\pi$ 的概率列。这里第一个固定点方程是 $0.9\pi_1+0.2\pi_2=\pi_1$，所以 $\pi_1=2\pi_2$；再归一化便唯一确定了所示 $\pi$。$P$ 的特征值是 $1$ 和 $0.7$，满足[[Markov矩阵幂收敛判据]]。极限正是 $L$：两个出发状态最终都给出 $2/3$、$1/3$ 的概率。

<!-- bilingual-en:start -->
## From convergent powers and uniqueness to the formula
<!-- bilingual-en:end -->

*由幂收敛和唯一性得到公式*

<!-- bilingual-en:start -->
Let $P$ be a finite real $n\times n$ column-stochastic matrix, with integer $n\ge1$. Thus $P_{ij}\ge0$ and $\sum_iP_{ij}=1$ for each column $j$. A probability column $p_0$ has nonnegative entries summing to one, and after $k$ steps it becomes $P^kp_0$. Every such matrix has [[有限Markov链稳态存在|at least one stationary distribution]]; here we additionally assume it is unique and that the entries of $P^k$ converge.
<!-- bilingual-en:end -->

设 $P$ 是有限实 $n\times n$ 列随机矩阵，整数 $n\ge1$，即 $P_{ij}\ge0$ 且每一列都满足 $\sum_iP_{ij}=1$。概率列 $p_0$ 的分量非负、总和为一，$k$ 步后变为 $P^kp_0$。这样的矩阵都[[有限Markov链稳态存在|至少有一个稳态分布]]；此处另外假设稳态唯一，并且 $P^k$ 的各条目收敛。

> [!important] One repeated column · 每一列都相同
> <!-- bilingual-en:start -->
> If the unique stationary column is $\pi$, then
>
> $$P^k\longrightarrow L=\pi\mathbf1^T,\qquad
> P^kp_0\longrightarrow\pi\quad\text{for every probability column }p_0.$$
>
> Here $\mathbf1=(1,\ldots,1)^T$ has $n$ entries. Multiplying the $n\times1$ column $\pi$ by the $1\times n$ row $\mathbf1^T$ produces an $n\times n$ matrix whose every column is $\pi$. Its [[矩阵秩|rank]] is one because its columns span the single nonzero direction $\pi$.
> <!-- bilingual-en:end -->
>
> 若唯一稳态列为 $\pi$，则 $P^k\to\pi\mathbf1^T$，且每个概率列初值都趋向 $\pi$。其中 $\mathbf1=(1,\ldots,1)^T$ 有 $n$ 个分量。$n\times1$ 的列 $\pi$ 乘 $1\times n$ 的行 $\mathbf1^T$，得到每列都是 $\pi$ 的 $n\times n$ 矩阵；各列只张成非零方向 $\pi$，所以[[矩阵秩|秩]]为一。

<!-- bilingual-en:start -->
The proof can be read column by column. Write $P^k\to L$. Each power has nonnegative entries and unit column sums, so every column of $L$ is a probability vector. Taking limits in $PP^k=P^{k+1}$ gives $PL=L$, which says that every column of $L$ is stationary. Uniqueness forces each of those columns to equal $\pi$. Finally,
<!-- bilingual-en:end -->

证明可以逐列理解。记 $P^k\to L$。每次幂都元素非负、每列之和为一，所以 $L$ 的每一列都是概率向量。对 $PP^k=P^{k+1}$ 取极限得到 $PL=L$，说明 $L$ 的每一列都平稳。唯一性迫使每列都等于 $\pi$。最后，

$$
Lp_0=\pi(\mathbf1^Tp_0)=\pi,
$$

<!-- bilingual-en:start -->
because $\mathbf1^Tp_0$ is the sum of the initial probabilities and equals one. The conclusion concerns the distribution of the state; it does not say that an individual random path eventually stops moving.
<!-- bilingual-en:end -->

因为 $\mathbf1^Tp_0$ 就是初始概率之和，等于一。结论描述的是状态的分布，并不是说某条随机路径最终会停止移动。

<!-- bilingual-en:start -->
## What the projection preserves
<!-- bilingual-en:end -->

*这个投影保留了什么*

<!-- bilingual-en:start -->
The [[Markov矩阵幂收敛判据|general convergence result]] identifies the range of $L$ with $E_1(P)=\{x\in\mathbb R^n:Px=x\}$, the fixed-point [[特征空间|eigenspace]]. Under the present assumptions this space is the line of multiples of $\pi$, so $\dim E_1(P)=1$. Conversely, a one-dimensional fixed-point space makes the stationary distribution unique after normalization; the [[Markov稳态唯一判据|uniqueness criterion]] proves that equivalence without assuming convergence. Thus, **once powers converge**, rank one, a one-dimensional fixed-point space, stationary uniqueness, and a common limit for all initial probability columns are equivalent descriptions.
<!-- bilingual-en:end -->

[[Markov矩阵幂收敛判据|一般收敛结论]]说明，$L$ 的值域就是固定点[[特征空间]] $E_1(P)=\{x\in\mathbb R^n:Px=x\}$。在当前条件下，这个空间是 $\pi$ 的全部倍数组成的直线，所以 $\dim E_1(P)=1$。反过来，一维固定点空间归一化后也只能给出一个稳态；[[Markov稳态唯一判据]]在不假设收敛时证明了这组等价。因此，**在幂已经收敛的前提下**，极限秩一、固定点空间一维、稳态唯一，以及全部概率初值具有共同极限，是等价的说法。

<!-- bilingual-en:start -->
For an arbitrary real or complex vector $x$, the formula is instead $Lx=\pi(\mathbf1^Tx)$. It preserves the total sum, which need not be one. In particular, a vector whose entries sum to zero is sent to zero. Since $\mathbf1^T\pi=1$, we have $L^2=L$: it is an [[幂等矩阵|idempotent projection]] onto the line through $\pi$. In the opening example, $L_{12}=2/3\ne1/3=L_{21}$, so $L^T\ne L$ and it is [[正交投影矩阵判别|not an orthogonal projection]] for the standard real inner product. A stationary limit need not be a perpendicular projection.
<!-- bilingual-en:end -->

对任意实向量或复向量 $x$，正确公式是 $Lx=\pi(\mathbf1^Tx)$；它保留分量总和，而这个总和未必为一。特别地，分量总和为零的向量会被送到零。因为 $\mathbf1^T\pi=1$，有 $L^2=L$，所以它是到 $\pi$ 所在直线的[[幂等矩阵|幂等投影]]。开头的例子满足 $L_{12}=2/3\ne1/3=L_{21}$，故 $L^T\ne L$，在标准实内积下[[正交投影矩阵判别|不是正交投影]]。稳态极限不必是垂直投影。

<!-- bilingual-en:start -->
## An exact formula for the two-state example
<!-- bilingual-en:end -->

*两状态例子的精确公式*

<!-- bilingual-en:start -->
For the displayed $P$ and $L$, direct calculation gives $P=0.7I_2+0.3L=L+0.7(I_2-L)$, where $I_2$ is the [[单位矩阵|identity]]. Because $L^2=L$, the products $L(I_2-L)$ and $(I_2-L)L$ vanish, while $I_2-L$ is also idempotent. Consequently, for every integer $k\ge0$,
<!-- bilingual-en:end -->

对上面的 $P,L$，直接计算得到 $P=0.7I_2+0.3L=L+0.7(I_2-L)$，其中 $I_2$ 是[[单位矩阵]]。由 $L^2=L$，交叉乘积 $L(I_2-L)$ 与 $(I_2-L)L$ 都为零，且 $I_2-L$ 也幂等。因此对每个整数 $k\ge0$，

$$
P^k=L+0.7^k(I_2-L),\qquad
P^kp_0=\pi+0.7^k(p_0-\pi).
$$

<!-- bilingual-en:start -->
The second formula assumes $p_0$ is a probability column. It shows exactly what decays: the difference from $\pi$ is multiplied by $0.7$ at every step. For $k=0$, the formulas give $P^0=I_2$ and the original $p_0$; as $k\to\infty$, the factor $0.7^k$ disappears. This simple scalar decay is specific to this example; larger decaying Jordan blocks may contribute polynomial factors in other matrices.
<!-- bilingual-en:end -->

第二个公式要求 $p_0$ 是概率列。它准确显示了衰减的对象：与 $\pi$ 的差每一步乘上 $0.7$。$k=0$ 时，公式给回 $P^0=I_2$ 和原始 $p_0$；$k\to\infty$ 时，$0.7^k$ 消失。这种单一标量衰减属于本例；其他矩阵中的较大衰减 Jordan 块还可能贡献多项式因子。

<!-- bilingual-en:start -->
## Rows, and the limits of the assumptions
<!-- bilingual-en:end -->

*换成概率行，以及条件的边界*

<!-- bilingual-en:start -->
For the same chain in the [[Markov矩阵左右约定|row convention]], put $R=P^T$. Then
<!-- bilingual-en:end -->

同一条链若采用[[Markov矩阵左右约定|行随机约定]]，令 $R=P^T$，则

$$
R^k\longrightarrow\mathbf1\pi^T,\qquad
r_0R^k\longrightarrow\pi^T
$$

<!-- bilingual-en:start -->
for every probability row $r_0$, since $r_0\mathbf1=1$. The all-ones vector is on the other side: repeated rows replace repeated columns.
<!-- bilingual-en:end -->

对每个概率行 $r_0$ 都成立，因为 $r_0\mathbf1=1$。全一向量换到另一侧，重复列也相应变成重复行。

> [!warning] Keep both assumptions · 两个条件都不能省
> <!-- bilingual-en:start -->
> $I_2^k=I_2$ converges but has many stationary distributions, so there is no common probability limit. The swap matrix $F=\begin{pmatrix}0&1\\1&0\end{pmatrix}$ has the unique stationary column $(1/2,1/2)^T$, but $F^{2k}=I_2$ and $F^{2k+1}=F$; [[稳态唯一不推收敛|uniqueness alone does not give convergence]]. Neither assumption requires all entries of $P$ or $\pi$ to be strictly positive: $A=\begin{pmatrix}1&1\\0&0\end{pmatrix}$ already equals its rank-one limit, with $\pi=(1,0)^T$. Irreducibility is not necessary for this theorem.
> <!-- bilingual-en:end -->
>
> $I_2^k=I_2$ 收敛，却有许多稳态，因此没有共同概率极限。交换矩阵 $F$ 的唯一稳态列为 $(1/2,1/2)^T$，但 $F^{2k}=I_2$、$F^{2k+1}=F$，说明[[稳态唯一不推收敛|唯一性本身不保证收敛]]。两个条件也都没有要求 $P$ 或 $\pi$ 的每个分量严格为正：$A=\begin{pmatrix}1&1\\0&0\end{pmatrix}$ 本身就等于其秩一极限，稳态是 $(1,0)^T$。本定理不以不可约为必要条件。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Gallager, *Discrete Stochastic Processes*, Chapter 3, Theorem 3.3.2 and §3.4.1, PDF pp.15–18 / printed pp.117–120](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/3558b08622765d26c2b0a7d2eeeac885_MIT6_262S11_chap03.pdf#page=15): the repeated-row limit for an ergodic finite unichain, including zero stationary probabilities at transient states, and the two-state power formula. The theorem here uses the explicit assumptions of convergent powers and unique stationarity, and proves the column version directly.
- [Levin–Peres with Wilmer, *Markov Chains and Mixing Times*, 2nd ed., §1.5.1 and Proposition 1.32, PDF pp.25–26 and 36](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf#page=25): the normalized stationary-distribution definition and finite-state existence. Existence alone is not used as a convergence assertion.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]], PDF pp.2–3, and [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S11_Lecture_Lecture_24_Markov_Matrices_Fourier_Series.pdf|Lecture 24 transcript]], PDF pp.8–14: the displayed two-state transition matrix, stationary proportions, and decaying mode. Its normalized probability formula, projection identities, and row/column conversion above are checked directly.
- [Nick Higham, *What Is a Stochastic Matrix?*](https://nhigham.com/2022/12/13/what-is-a-stochastic-a-matrix/): supplementary examples of convergent and alternating stochastic powers. The column-by-column argument above supplies the rank-one proof.
<!-- bilingual-en:end -->

- Gallager 第3章定理3.3.2及 §3.4.1，PDF第15–18页 / 印刷第117–120页：有限遍历单常返类链的重复行极限，包括暂态位置的稳态概率为零，以及两状态幂公式。本文直接假设幂收敛且稳态唯一，并逐列证明对应结论。
- Levin–Peres 与 Wilmer，第2版 §1.5.1及命题1.32，PDF第25–26、36页：归一化稳态定义和有限状态存在性。存在性不被当作收敛断言。
- MIT Session 2.11摘要第2–3页、Lecture 24讲稿第8–14页：本卡两状态矩阵、稳态比例和衰减模式。其归一化概率公式、投影等式和行列转换直接核验。
- Higham 文章提供幂收敛和交替振荡的补充例子；秩一结论由本文逐列论证证明。

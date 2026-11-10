---
aliases:
  - Existence of stationary distributions for finite Markov chains
  - Finite-state stationary distribution existence
  - 有限状态链的稳态存在性
student_os: knowledge-atom
atom_id: LA-EIG-051
atom_type: theorem
status: source-checked
requires:
  - "[[Markov稳态分布]]"
  - "[[紧致性的序列判别]]"
related:
  - "[[Markov稳态唯一判据]]"
  - "[[Markov矩阵幂收敛判据]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
  - "[[离散时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# Stationary distributions exist for finite Markov chains
<!-- bilingual-en:end -->

*有限 Markov 链的稳态存在性*

<!-- bilingual-en:start -->
A chain can keep moving forever and still admit a probability allocation that its transitions leave unchanged. For a **finite, nonempty state space**, such an allocation always exists. This theorem guarantees a [[Markov稳态分布|stationary distribution]] without assuming that every state can reach every other state or that the distributions at successive times converge.
<!-- bilingual-en:end -->

一条链可以一直运动，同时仍存在一种转移前后保持不变的概率分配。只要状态空间**有限且非空**，这样的分配就一定存在。这个定理保证[[Markov稳态分布]]的存在，不要求各状态彼此都能到达，也不要求逐时刻的分布收敛。

> [!example] Averaging a two-state cycle · 给两状态循环取平均
> <!-- bilingual-en:start -->
> Let $F=\begin{pmatrix}0&1\\1&0\end{pmatrix}$ and start with the probability column $p=(1,0)^T$, where $T$ means [[矩阵转置|transpose]]. The successive columns are $(1,0)^T,(0,1)^T,(1,0)^T,\ldots$. They do not converge. But the average of the first two, four or six columns is always $(1/2,1/2)^T$, which satisfies $F(1/2,1/2)^T=(1/2,1/2)^T$. Time averages suggest a way to find a stationary distribution even when individual times oscillate.
> <!-- bilingual-en:end -->
>
> 令 $F=\begin{pmatrix}0&1\\1&0\end{pmatrix}$，从概率列 $p=(1,0)^T$ 出发，$T$ 表示[[矩阵转置|转置]]。逐时刻的列在 $(1,0)^T$ 与 $(0,1)^T$ 之间交替，不会收敛；但前二、四或六个分布的平均总是 $(1/2,1/2)^T$，而这个平均满足固定点方程。即使单个时刻振荡，时间平均仍提示了寻找稳态的办法。

<!-- bilingual-en:start -->
## The existence statement
<!-- bilingual-en:end -->

*定理保证什么*

<!-- bilingual-en:start -->
Let $P$ be a fixed real $n\times n$ [[Markov矩阵|column-stochastic matrix]], with an integer $n\ge1$: its entries are nonnegative and each column sums to one. Then there is a column $\pi$ such that

$$P\pi=\pi,\qquad \pi_i\ge0\ (1\le i\le n),\qquad \sum_{i=1}^n\pi_i=1.$$

The last two conditions make $\pi$ a probability distribution, rather than merely an eigenvector. In the [[Markov矩阵左右约定|row convention]], the same theorem gives a probability row $\pi^T$ with $\pi^TP=\pi^T$ for a row-stochastic $P$. The proof below uses columns throughout.
<!-- bilingual-en:end -->

设 $P$ 为固定的实 $n\times n$ [[Markov矩阵|列随机矩阵]]，整数 $n\ge1$：元素非负，每列和为一。那么存在列 $\pi$ 同时满足上面三个条件。后两个条件确保 $\pi$ 是概率分布，而不只是一个特征向量。采用[[Markov矩阵左右约定|行约定]]时，定理等价地保证行随机矩阵 $P$ 存在概率行 $\pi^T$，满足 $\pi^TP=\pi^T$。以下证明始终采用概率列。

<!-- bilingual-en:start -->
## Average the distributions, then take a limit
<!-- bilingual-en:end -->

*先平均分布，再取极限*

<!-- bilingual-en:start -->
Every finite stochastic matrix has at least one stationary distribution, even if some states cannot reach each other or the chain cycles. One way to see this is to average distributions over time. Start from any probability column $p$ and let $N\ge1$ be an integer. Define

$$q_N=\frac1N\sum_{k=0}^{N-1}P^kp.$$

Each $q_N$ is still nonnegative and has sum one. Applying one extra transition makes nearly every term cancel:

$$Pq_N-q_N=\frac{P^Np-p}{N},\qquad
\|Pq_N-q_N\|_1\le\frac2N.$$

The notation $\|z\|_1=\sum_i|z_i|$ adds coordinate magnitudes. The bound is two divided by $N$ because both $P^Np$ and $p$ have absolute sum one. The time average therefore becomes closer and closer to satisfying the fixed-point equation.
<!-- bilingual-en:end -->

每个有限随机矩阵都至少有一个稳态分布，即使某些状态互相到不了，或状态变化存在循环。可以通过对不同时刻的分布取平均来理解。任选概率列 $p$，对整数 $N\ge1$，令 $q_N$ 为前 $N$ 个分布 $p,Pp,\ldots,P^{N-1}p$ 的平均。每个 $q_N$ 仍非负、总和一。再转移一步后，几乎全部中间项相消，得到上面的 $Pq_N-q_N=(P^Np-p)/N$。记号 $\|z\|_1=\sum_i|z_i|$ 把坐标的模相加；因 $P^Np$ 和 $p$ 的绝对值和都为一，差的大小至多为 $2/N$。所以这个时间平均越来越接近固定点方程。

<!-- bilingual-en:start -->
To obtain an exact fixed point, use one finite-dimensional fact. The set of all probability columns is closed and bounded in $\mathbb R^n$, hence [[Heine–Borel 定理的边界|compact]]. Its [[紧致性的序列判别|sequential form]] means that some subsequence of the $q_N$ converges to a probability column $\pi$ still inside the set. Along that subsequence, the displayed error tends to zero; matrix multiplication consists of finite sums, so passing to the limit gives $P\pi=\pi$. This proves existence. It has not required the individual distributions $P^kp$ to converge.
<!-- bilingual-en:end -->

要得到一个精确固定点，用到一个有限维事实：全部概率列在 $\mathbb R^n$ 中构成闭且有界集合，因此由[[Heine–Borel 定理的边界|Heine–Borel 判别]]是紧致的。[[紧致性的序列判别]]表示，$q_N$ 中总能选出一条收敛子序列，极限 $\pi$ 仍是集合内的概率列。沿这条子序列，上面的误差趋零；矩阵乘法只有有限个加法项，因此取极限得到 $P\pi=\pi$。这就证明了存在性，而没有要求单个时刻的分布 $P^kp$ 收敛。

<!-- bilingual-en:start -->
The proof only needs one convergent subsequence of the averages; it does not assert convergence of the unaveraged sequence $P^kp$. Nor does it establish uniqueness: for the [[单位矩阵|identity matrix]] $I_n$, every probability column is stationary. The [[Markov稳态唯一判据|uniqueness criterion]] and [[Markov矩阵幂收敛判据|convergence criterion]] answer those separate questions.
<!-- bilingual-en:end -->

这个证明只需要平均序列中存在一条收敛子序列，没有断言未取平均的 $P^kp$ 收敛。它也不保证唯一：对[[单位矩阵]] $I_n$，每个概率列都平稳。[[Markov稳态唯一判据]]和[[Markov矩阵幂收敛判据]]分别回答另外两个问题。

> [!warning] Finiteness keeps probability from escaping · 有限性防止概率逃向无穷
> <!-- bilingual-en:start -->
> On states $0,1,2,\ldots$, suppose the chain always moves from $j$ to $j+1$. A stationary distribution would need $\pi_0=0$, since nothing enters state zero, and $\pi_j=\pi_{j-1}$ for every $j\ge1$. All entries would be zero, contradicting total mass one. Starting at zero, the average of the first $N$ distributions assigns mass $1/N$ to each of $0,\ldots,N-1$. Every fixed coordinate tends to zero, but this coordinatewise limit is not a probability distribution. Thus the finite-state compactness step cannot simply be carried over to infinitely many states.
> <!-- bilingual-en:end -->
>
> 在状态 $0,1,2,\ldots$ 上，假设每次都从 $j$ 移到 $j+1$。稳态必须满足 $\pi_0=0$，因为没有概率流入零；还须对全部 $j\ge1$ 满足 $\pi_j=\pi_{j-1}$。于是每项都为零，与总和一矛盾。从零出发，前 $N$ 个分布的平均在 $0,\ldots,N-1$ 各放质量 $1/N$。每个固定坐标都趋于零，但这个逐坐标极限不是概率分布。因此，有限状态下的紧致性步骤不能直接搬到无限状态空间。

## 来源与核验

<!-- bilingual-en:start -->
- [Levin–Peres with Wilmer, Markov Chains and Mixing Times, 2nd ed., Proposition 1.32, PDF p.36 / printed p.20](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf#page=36): existence for every finite stochastic matrix, including reducible matrices, by averaging powers and compactness. The book uses rows. The column proof here uses the existence part of the proposition, not its additional conclusion about convergence of the whole sequence of averages. The swap, identity and one-way infinite chain examples are verified directly.
<!-- bilingual-en:end -->

- Levin–Peres 等第2版，命题1.32，PDF第36页／印刷20页：核对全部有限随机矩阵（包括可约矩阵）通过幂平均和紧致性得到稳态的存在性。原书使用行，本文转为列，并只使用存在性部分，没有沿用该命题对整个平均序列收敛的额外结论。交换矩阵、单位矩阵及无限单向链例子均直接核算。

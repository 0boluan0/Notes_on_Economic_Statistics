---
aliases:
  - Markov matrix
  - Stochastic matrix
student_os: knowledge-atom
atom_id: LA-EIG-039
atom_set: eigenvalues-linear-dynamics
atom_type: definition
status: source-checked
related:
  - "[[Markov矩阵左右约定]]"
  - "[[Markov矩阵必有特征值一]]"
  - "[[Markov矩阵谱边界]]"
  - "[[DTMC时间齐次转移核]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

<!-- bilingual-en:start -->
# Markov matrices
<!-- bilingual-en:end -->

*Markov 矩阵：把当前概率分配到下一步*

<!-- bilingual-en:start -->
A **Markov matrix**, also called a **stochastic matrix**, stores a rule for moving among a finite set of states. For each possible current state, it lists the probabilities of the next state. Multiplying by this matrix combines those rules with our current probabilities to produce the next distribution.
<!-- bilingual-en:end -->

**Markov 矩阵**，也叫**随机矩阵**，把有限个状态之间的转移规则写成数表：对每个可能的当前状态，列出下一步到各状态的概率。把矩阵与当前概率相乘，就能把这些转移规则合起来，得到下一步的分布。

<!-- bilingual-en:start -->
## Two states, one step
<!-- bilingual-en:end -->

*先看两个状态的一步转移*

<!-- bilingual-en:start -->
Suppose the states are $A$ and $B$. Starting from $A$, the next state is $A$ with probability $0.9$ and $B$ with probability $0.1$. Starting from $B$, the corresponding probabilities are $0.2$ and $0.8$. Put the probabilities for each **starting state in a column**:
<!-- bilingual-en:end -->

设状态是 $A$ 与 $B$。从 $A$ 出发，下一步以 $0.9$ 的概率到 $A$、以 $0.1$ 的概率到 $B$；从 $B$ 出发，对应概率为 $0.2$ 和 $0.8$。把**同一个出发状态的概率放在一列**，得到

$$
P=\begin{bmatrix}0.9&0.2\\0.1&0.8\end{bmatrix}.
$$

<!-- bilingual-en:start -->
The first column describes departures from $A$, and the second describes departures from $B$. Within each column, the first row means arrival at $A$ and the second means arrival at $B$. Each column sums to $1$, because these are all the possible destinations.
<!-- bilingual-en:end -->

第一列描述从 $A$ 出发，第二列描述从 $B$ 出发；每一列内部，第一行表示到达 $A$，第二行表示到达 $B$。每列加起来为 $1$，因为已经列出全部可能的去处。

<!-- bilingual-en:start -->
A **probability vector** lists nonnegative probabilities whose sum is $1$; it is a finite [[概率质量函数|probability mass function]] written as a vector after numbering the states. If the current probabilities of $A,B$ are $0.6,0.4$, write them as the column $x=(0.6,0.4)^T$. Here $T$ denotes [[矩阵转置|transpose]], turning the horizontal list into a column. [[矩阵乘法|Matrix multiplication]] gives
<!-- bilingual-en:end -->

**概率向量**把非负且总和为 $1$ 的概率排成一个向量；给状态编号后，它就是有限[[概率质量函数]]的向量写法。若当前位于 $A,B$ 的概率分别为 $0.6,0.4$，把它们写成列 $x=(0.6,0.4)^T$。上标 $T$ 表示[[矩阵转置]]，把横写的坐标变成列。由[[矩阵乘法]]得到

$$
x'=Px
=\begin{bmatrix}0.9(0.6)+0.2(0.4)\\0.1(0.6)+0.8(0.4)\end{bmatrix}
=\begin{bmatrix}0.62\\0.38\end{bmatrix}.
$$

<!-- bilingual-en:start -->
Here $x'$ means the next probability vector, not a derivative. The probability $0.62$ of arriving at $A$ combines two routes: start at $A$ and stay, contributing $0.6(0.9)$; or start at $B$ and move to $A$, contributing $0.4(0.2)$. The new probabilities remain nonnegative and sum to $1$.
<!-- bilingual-en:end -->

这里 $x'$ 表示下一步概率向量，不是导数。到达 $A$ 的概率 $0.62$ 合并了两条路线：先在 $A$ 再留下，贡献 $0.6(0.9)$；或先在 $B$ 再到 $A$，贡献 $0.4(0.2)$。新的概率仍然非负且总和为 $1$。

<!-- bilingual-en:start -->
## The two normalization conventions
<!-- bilingual-en:end -->

*行随机与列随机的定义*

<!-- bilingual-en:start -->
Let $P=(p_{ij})$ be a real $n\times n$ matrix, with a positive integer $n$ states. The entry $p_{ij}$ lies in row $i$, column $j$. Both conventions require **every entry to be nonnegative**. They differ in which sets of entries must sum to $1$:
<!-- bilingual-en:end -->

设 $P=(p_{ij})$ 是实 $n\times n$ 方阵，状态数 $n$ 是正整数；$p_{ij}$ 位于第 $i$ 行、第 $j$ 列。两种约定都要求**每个元素非负**，区别在于哪一组元素的和必须为 $1$：

$$
\begin{aligned}
\text{column-stochastic:}&\quad p_{ij}\ge0,\quad \sum_{i=1}^{n}p_{ij}=1\quad\text{for each column }j;\\
\text{row-stochastic:}&\quad p_{ij}\ge0,\quad \sum_{j=1}^{n}p_{ij}=1\quad\text{for each row }i.
\end{aligned}
$$

<!-- bilingual-en:start -->
The summation symbol means to add over the indicated index. Column-stochastic means that each **column** is a probability vector; row-stochastic means that each **row** is one. A matrix satisfying both is called **doubly stochastic**. In our example the row sums are $1.1$ and $0.9$, so $P$ is column-stochastic but not row-stochastic.
<!-- bilingual-en:end -->

求和符号表示沿指定下标把元素相加。列随机就是每**列**分别构成一个概率向量；行随机就是每**行**分别构成一个概率向量。同时满足两种要求的矩阵称为**双随机矩阵**。上例的行和为 $1.1$ 与 $0.9$，所以 $P$ 是列随机矩阵，但不是行随机矩阵。

<!-- bilingual-en:start -->
To describe the same transitions with rows, transpose the matrix and multiply a row probability vector on the right. Thus the two conventions agree after the matrix and vector orientations are changed together; [[Markov矩阵左右约定|row and column conventions]] explains the entry indices, multiplication order, and fixed-point equations. MIT 18.06 uses the column convention.
<!-- bilingual-en:end -->

若想用行描述相同的转移，就把矩阵转置，让概率行向量从左边乘它。因此，两种约定只要同时调整矩阵与向量的方向，就能描述同一个过程；[[Markov矩阵左右约定]]具体解释下标、乘法顺序与固定点方程。MIT 18.06 使用列随机约定。

> [!warning] Unit sums alone do not give probabilities · 和为一还不够
> <!-- bilingual-en:start -->
> The matrix $H=\begin{bmatrix}1.1&0\\-0.1&1\end{bmatrix}$ has unit column sums, but sends the valid probability vector $(1,0)^T$ to $(1.1,-0.1)^T$. The negative component makes it invalid as a Markov matrix. Zero entries, however, are allowed: the [[单位矩阵|identity matrix]] represents staying in the same state with probability one.
> <!-- bilingual-en:end -->
>
> 矩阵 $H=\begin{bmatrix}1.1&0\\-0.1&1\end{bmatrix}$ 的每列之和为一，但会把合法概率向量 $(1,0)^T$ 变成 $(1.1,-0.1)^T$。负分量使它不符合 Markov 矩阵的要求。零元素则完全允许：[[单位矩阵]]表示以概率一留在原状态。

<!-- bilingual-en:start -->
A single matrix specifies one transition rule. Reusing that same rule at every step is the additional [[DTMC时间齐次转移核|time-homogeneous Markov-chain model]]. The matrix definition alone does not establish that observed data follow that model, nor does it assert convergence to a unique long-run distribution.
<!-- bilingual-en:end -->

一个矩阵给出一套转移规则。每一步都重复使用同一规则，对应额外的[[DTMC时间齐次转移核|时间齐次 Markov 链模型]]。矩阵满足定义，本身不能证明观测数据服从该模型，也没有断言分布最终收敛到唯一极限。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Levin and Peres, with Wilmer, *Markov Chains and Mixing Times*, 2nd ed., §1.1](https://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf#page=18), printed pp.2–5 / PDF pp.18–21: stochastic rows, probability distributions, and the one-step weighted update.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]], pp.1–3: the course column convention and the displayed two-state matrix. Every displayed update is calculated directly.
- [Higham, *What Is a Stochastic Matrix?*](https://nhigham.com/2022/12/13/what-is-a-stochastic-a-matrix/): supplementary terminology for row-, column-, and doubly stochastic matrices.
<!-- bilingual-en:end -->

- Levin、Peres 与 Wilmer，《Markov Chains and Mixing Times》第2版 §1.1，印刷第2–5页 / PDF第18–21页：支持行随机定义、概率分布与一步加权更新。
- MIT 18.06SC Session 2.11 summary，第1–3页：支持课程的列随机约定与正文两状态矩阵；展示的更新直接计算。
- Higham《What Is a Stochastic Matrix?》：补充行随机、列随机与双随机的术语。

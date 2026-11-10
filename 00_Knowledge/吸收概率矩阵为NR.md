---
aliases:
- Absorption-probability matrix
- Absorption probabilities via the fundamental matrix
- 吸收概率矩阵
student_os: knowledge-atom
atom_id: PROB-DTMC-038
atom_set: discrete-time-markov-chains
atom_type: formula
status: source-checked
requires:
- '[[吸收链基本矩阵]]'
- '[[吸收链标准形]]'
related:
- '[[命中概率第一步方程]]'
leads_to: []
part_of:
- '[[离散时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Absorption probabilities
<!-- bilingual-en:end -->

*吸收概率矩阵*

<!-- bilingual-en:start -->
A system can spend several steps in intermediate states before ending in one of several permanent outcomes. The **absorption-probability matrix** answers “where will it eventually end?” for every intermediate starting state. It includes all possible lengths of the journey, not just the next step.
<!-- bilingual-en:end -->

一个系统可能在中间状态停留多步，最后才进入某个永久结果。**吸收概率矩阵**为每个中间起点回答“最终会停在哪里”，把各种长短的路径都计入，而不只看下一步。

<!-- bilingual-en:start -->
## Two stages and two possible endings
<!-- bilingual-en:end -->

*两个阶段与两个终点*

<!-- bilingual-en:start -->
Consider a process with intermediate states $u,v$ and permanent outcomes $a,b$. From $u$, stay at $u$ with probability $1/2$, move to $v$ with probability $1/4$, and finish at $a$ with probability $1/4$. From $v$, stay there with probability $1/2$, or finish at $a$ or $b$ with probability $1/4$ each. The outcomes $a,b$ are [[吸收状态|absorbing states]]: once reached, they are never left.
<!-- bilingual-en:end -->

考虑中间状态 $u,v$ 和永久终点 $a,b$。从 $u$ 出发，以 $1/2$ 的概率留在 $u$，以 $1/4$ 到 $v$，以 $1/4$ 在 $a$ 结束。从 $v$ 出发，以 $1/2$ 留在原处，以各 $1/4$ 在 $a$ 或 $b$ 结束。$a,b$ 都是[[吸收状态]]：一旦进入，就不再离开。

<!-- bilingual-en:start -->
The one-step probability of going directly from $u$ to $b$ is zero, but the eventual probability is positive: a path can first reach $v$. Separating intermediate moves from final exits gives the two blocks
<!-- bilingual-en:end -->

从 $u$ 下一步直接到 $b$ 的概率为零，但最终到达 $b$ 的概率为正，因为可以先到 $v$。把中间转移与最终出口分开，得到两个矩阵块：

$$
Q=\begin{pmatrix}1/2&1/4\\0&1/2\end{pmatrix},\qquad
R=\begin{pmatrix}1/4&0\\1/4&1/4\end{pmatrix}.
$$

<!-- bilingual-en:start -->
The rows of both blocks are ordered $u,v$. The columns of $Q$ are $u,v$, while the columns of $R$ are $a,b$. Thus $Q$ tracks moves that stay in the intermediate region, and $R$ tracks a one-step exit. These are the [[吸收链标准形|canonical absorbing-chain blocks]].
<!-- bilingual-en:end -->

两个块的行都按 $u,v$ 排列；$Q$ 的列是 $u,v$，$R$ 的列是 $a,b$。所以 $Q$ 描述仍留在中间区域的转移，$R$ 描述一步离开。这就是[[吸收链标准形]]中的两个块。

<!-- bilingual-en:start -->
## What the formula assumes and returns
<!-- bilingual-en:end -->

*公式的条件与输出*

<!-- bilingual-en:start -->
More generally, take a finite, time-homogeneous [[吸收链|absorbing Markov chain]]: at least one state is absorbing, and every other state can reach an absorbing state along a path with positive probability. Use the [[Markov矩阵左右约定|row convention]], with $m\ge1$ nonabsorbing states first and $r\ge1$ absorbing states last. The former are transient in this finite setting: they are eventually left for the absorbing region with probability one, as proved in [[吸收链必然吸收|eventual absorption]]. Write
<!-- bilingual-en:end -->

更一般地，取有限、时间齐次的[[吸收链]]：至少存在一个吸收状态，每个其他状态都能沿正概率路径到达某个吸收状态。采用[[Markov矩阵左右约定|行约定]]，先列 $m\ge1$ 个非吸收状态，再列 $r\ge1$ 个吸收状态。在这个有限情形中，前者确实是暂态状态：最终以概率一离开它们并进入吸收区域，证明见[[吸收链必然吸收]]。写成

$$
P=\begin{pmatrix}Q&R\\0&I_r\end{pmatrix},\qquad
N=(I_m-Q)^{-1}=\sum_{k=0}^{\infty}Q^k.
$$

<!-- bilingual-en:start -->
Here $Q$ is $m\times m$, $R$ is $m\times r$, and $I_m,I_r$ are [[单位矩阵|identity matrices]] of the indicated sizes. The [[吸收链基本矩阵|fundamental matrix]] $N$ records expected visits: $N_{ij}$ is the average number of visits to transient state $j$ before absorption, starting at transient state $i$, including time zero. Its finite entries and inverse formula follow from the [[基本矩阵存在条件|transient-block conditions]].
<!-- bilingual-en:end -->

这里 $Q$ 为 $m\times m$，$R$ 为 $m\times r$，$I_m,I_r$ 是相应大小的[[单位矩阵]]。[[吸收链基本矩阵|基本矩阵]] $N$ 记录期望访问次数：$N_{ij}$ 是从暂态状态 $i$ 出发、吸收前访问暂态状态 $j$ 的平均总次数，包括时刻零。其元素有限及求逆公式由[[基本矩阵存在条件|暂态块条件]]保证。

> [!important] All eventual destinations at once · 同时求出全部终点概率
> <!-- bilingual-en:start -->
> Let $B_{ia}$ be the probability of eventual absorption at absorbing state $a$, starting at transient state $i$. Then $B$ is an $m\times r$ matrix and
> <!-- bilingual-en:end -->
>
> 令 $B_{ia}$ 是从暂态状态 $i$ 出发、最终被吸收到状态 $a$ 的概率。$B$ 是 $m\times r$ 矩阵，并且
>
> $$B=NR,\qquad B_{ia}=\sum_{j=1}^{m}N_{ij}R_{ja}.$$
>
> <!-- bilingual-en:start -->
> In this [[矩阵乘法|matrix product]], a row of expected visits is paired with a column of exit probabilities. $R$ describes a single next step; $B$ describes the final outcome after any number of steps.
> <!-- bilingual-en:end -->
>
> 这个[[矩阵乘法]]把一行期望访问次数与一列出口概率配对。$R$ 只描述下一步；$B$ 描述经过任意多步以后的最终结果。

<!-- bilingual-en:start -->
## Why expected visits become a probability
<!-- bilingual-en:end -->

*为什么期望访问次数能给出概率*

<!-- bilingual-en:start -->
For each transient state $j$, count transitions from $j$ into a particular absorbing state $a$. At any fixed time, conditional on being at $j$, the [[Markov性|Markov property]] gives probability $R_{ja}$ of that next transition. Summing over times makes the expected count equal to $N_{ij}R_{ja}$. This is an expectation calculation, not an assumption that successive visits are independent.
<!-- bilingual-en:end -->

对每个暂态状态 $j$，数从 $j$ 进入某个指定吸收状态 $a$ 的转移。在每个固定时刻，给定当前在 $j$，[[Markov性]]使下一步发生该转移的概率为 $R_{ja}$。对时刻求和后，期望次数就是 $N_{ij}R_{ja}$。这里使用的是期望计算，并未假设历次访问相互独立。

<!-- bilingual-en:start -->
Across all $j$, such an entrance happens exactly once if the path ends at $a$, and zero times otherwise. Its total count is therefore an [[事件指示变量|indicator]], whose expectation is the probability of ending at $a$. Nonnegative sums may be passed through expectation by [[单调收敛定理|monotone convergence]]. This gives the formula above.
<!-- bilingual-en:end -->

再对所有 $j$ 求和：若路径最终停在 $a$，进入 $a$ 的那次转移恰好发生一次；否则发生零次。因此总次数是一个[[事件指示变量|指示变量]]，它的期望正是最终到达 $a$ 的概率。非负求和可以借助[[单调收敛定理]]移到期望外面，于是得到上面的公式。

<!-- bilingual-en:start -->
The [[命中概率第一步方程|first-step equations]] give the same result more compactly. Either the next step finishes at an absorbing state, contributing $R$, or it reaches another transient state and continues with probabilities $B$. Thus
<!-- bilingual-en:end -->

[[命中概率第一步方程]]给出同一个结果的简洁写法：下一步要么直接进入吸收状态，贡献 $R$；要么进入另一个暂态状态，再按 $B$ 中的概率继续。因此

$$B=R+QB,\qquad (I_m-Q)B=R,\qquad B=NR.$$

> [!example] Read one row as a distribution · 把一行读成一个分布
> <!-- bilingual-en:start -->
> For the two-stage example, the fundamental matrix is $N=\begin{pmatrix}2&1\\0&2\end{pmatrix}$. Therefore
> <!-- bilingual-en:end -->
>
> 在两个阶段的例子中，基本矩阵为 $N=\begin{pmatrix}2&1\\0&2\end{pmatrix}$。于是
>
> $$
> B=\begin{pmatrix}2&1\\0&2\end{pmatrix}
>   \begin{pmatrix}1/4&0\\1/4&1/4\end{pmatrix}
>  =\begin{pmatrix}3/4&1/4\\1/2&1/2\end{pmatrix}.
> $$
>
> <!-- bilingual-en:start -->
> Starting at $u$, eventual absorption at $a$ has probability $2(1/4)+1(1/4)=3/4$, and at $b$ probability $1/4$. Starting at $v$, the two endings are equally likely. In particular, the positive $B_{ub}$ includes the indirect route through $v$, which the zero entry $R_{ub}$ alone misses.
> <!-- bilingual-en:end -->
>
> 从 $u$ 出发，最终到 $a$ 的概率是 $2(1/4)+1(1/4)=3/4$，到 $b$ 的概率是 $1/4$；从 $v$ 出发，两个终点等可能。特别是，正数 $B_{ub}$ 包含了经 $v$ 的间接路径，而仅看零元素 $R_{ub}$ 会漏掉它。

<!-- bilingual-en:start -->
Each row of $B$ sums to one because eventual absorption occurs almost surely and the destinations are mutually exclusive. Algebraically, let $\mathbf1_m,\mathbf1_r$ be all-ones column vectors. The rows of $P$ sum to one, so $R\mathbf1_r=(I_m-Q)\mathbf1_m$, giving $B\mathbf1_r=\mathbf1_m$. The rows of $N$ do not generally sum to one: their sums answer a different question, [[期望吸收时间为N1|expected time to absorption]].
<!-- bilingual-en:end -->

$B$ 每行和为一，因为最终必然吸收，且各终点互斥。代数上，令 $\mathbf1_m,\mathbf1_r$ 为全一列向量；$P$ 的行和为一，所以 $R\mathbf1_r=(I_m-Q)\mathbf1_m$，从而 $B\mathbf1_r=\mathbf1_m$。$N$ 的行和一般不是一：它回答另一个问题，即[[期望吸收时间为N1|期望吸收时间]]。

<!-- bilingual-en:start -->
## A one-state check and the boundary of use
<!-- bilingual-en:end -->

*单状态核对与适用边界*

<!-- bilingual-en:start -->
With one transient state, let its probability of staying be $0\le q<1$, and its two exit probabilities be $r_1,r_2\ge0$, with $q+r_1+r_2=1$. Repeated waiting changes the time to exit, while the eventual destination shares are
<!-- bilingual-en:end -->

只有一个暂态状态时，设留下的概率为 $0\le q<1$，两个出口概率为 $r_1,r_2\ge0$，且 $q+r_1+r_2=1$。反复等待影响退出时间，而最终两个出口的占比为

$$B=\frac{1}{1-q}(r_1\quad r_2)
=\left(\frac{r_1}{r_1+r_2}\quad\frac{r_2}{r_1+r_2}\right).$$

<!-- bilingual-en:start -->
If we start at an absorbing state $a$, the final destination is already $a$ with probability one; that starting case is outside the transient rows of $B$. If all states are absorbing, there are no transient rows to calculate. Conversely, merely having some absorbing states does not justify the formula: a nonabsorbing [[闭沟通类|closed communicating class]] can trap the chain elsewhere. Then the alleged nonabsorbing block is not transient, $I-Q$ can be singular, and the true hitting probabilities must be handled with the appropriate boundary equations.
<!-- bilingual-en:end -->

若起点本来就是吸收状态 $a$，则最终终点已经以概率一确定为 $a$；这类起点不属于 $B$ 的暂态行。若所有状态都吸收，就没有需要计算的暂态行。反过来，仅仅存在一些吸收状态并不足以使用公式：一个非吸收的[[闭沟通类]]可能把链困在别处。此时所谓非吸收块不是暂态块，$I-Q$ 可能奇异，真实命中概率须按恰当的边界方程处理。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Grinstead and Snell, Introduction to Probability, §11.2, Theorems 11.3–11.6, PDF pp. 425–429 / printed pp. 417–421](https://math.dartmouth.edu/~prob/prob/prob.pdf#page=428): finite absorbing-chain assumptions, the fundamental-matrix visit interpretation, and Theorem 11.6 with its proof of $B=NR$ on PDF p. 428. The two-stage example and row-sum calculation are worked out directly here.
- [[01_Math/05_随机过程/04_马尔可夫链#4.3.3 吸收概率与吸收次数|Course notes, §4.3.3]]: the local notation $F=WR$ corresponds to $B=NR$ here.
<!-- bilingual-en:end -->

- Grinstead–Snell 第 11.2 节、定理 11.3–11.6（PDF 第 425–429 页，印刷第 417–421 页）支持有限吸收链条件、基本矩阵的访问解释，以及 PDF 第 428 页定理 11.6 的 $B=NR$ 与证明。两阶段算例与行和在正文直接计算。
- 课程第 4.3.3 节用 $F=WR$，对应本文的 $B=NR$。

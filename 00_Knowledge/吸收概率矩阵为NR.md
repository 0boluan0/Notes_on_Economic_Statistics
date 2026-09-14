---
aliases:
  - "吸收概率矩阵等于基本矩阵乘以暂态到吸收态的转移块"
  - Absorption probabilities in a finite absorbing chain equal NR
  - B equals NR for a finite absorbing chain
student_os: knowledge-atom
atom_id: PROB-DTMC-038
atom_set: discrete-time-markov-chains
atom_type: formula
status: source-checked
mastery_state: unassessed
requires:
  - "[[吸收链基本矩阵]]"
  - "[[吸收链标准形]]"
related:
  - "[[命中概率第一步方程]]"
leads_to: []
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# 吸收概率矩阵等于基本矩阵乘以暂态到吸收态的转移块
<!-- bilingual-en:start -->
*The absorption-probability matrix is the fundamental matrix times the transient-to-absorbing block*
<!-- bilingual-en:end -->

> [!summary] $B=NR$
> 对有限吸收链的标准形
> $$P=\begin{pmatrix}Q&R\\0&I\end{pmatrix}$$
> 以及基本矩阵 $N=(I-Q)^{-1}$，令 $B_{ia}$ 表示从暂态状态 $i$ 出发最终被吸收到状态 $a$ 的概率。则
> $$B=NR.$$
> <!-- bilingual-en:start -->
> For a finite absorbing chain in canonical form, let $B_{ia}$ be the probability of eventual absorption in state $a$ from transient state $i$. With $N=(I-Q)^{-1}$, the whole matrix is $B=NR$.
> <!-- bilingual-en:end -->

$N_{ij}$ 给出吸收前访问暂态状态 $j$ 的期望次数，每次位于 $j$ 时，下一步以概率 $R_{ja}$ 进入吸收状态 $a$。因为进入 $a$ 后不会离开，这类转移的期望次数就是最终被 $a$ 吸收的概率。对所有暂态状态 $j$ 求和，正好得到 $(NR)_{ia}$。
<!-- bilingual-en:start -->
The entry $N_{ij}$ is the expected number of visits to transient state $j$ before absorption. On each such visit, the next transition enters absorbing state $a$ with probability $R_{ja}$. Because absorption in $a$ can occur only once, the expected number of these transitions is also their probability; summing over $j$ gives $(NR)_{ia}$.
<!-- bilingual-en:end -->

这个公式依赖 [[吸收链标准形]] 和 [[吸收链基本矩阵]] 的存在条件。如果所谓“非吸收块”中仍含有不能离开的闭类，那么 $Q$ 就不是暂态块，不能直接套用 $B=NR$；应先重新分解状态空间。
<!-- bilingual-en:start -->
The formula requires the canonical absorbing-chain decomposition and a well-defined fundamental matrix. If the alleged nonabsorbing block still contains a closed class, then $Q$ is not transient and $B=NR$ cannot be used before the state space is decomposed correctly.
<!-- bilingual-en:end -->

若只有一个暂态状态，$Q=(q)$、$R=(r_1\;r_2)$ 且 $q+r_1+r_2=1$，则
$$
B=\frac{1}{1-q}(r_1\;r_2)
=\left(\frac{r_1}{r_1+r_2},\frac{r_2}{r_1+r_2}\right).
$$
<!-- bilingual-en:start -->
With one transient state, repeated returns contribute the geometric factor $(1-q)^{-1}$. The two entries reduce to the relative shares of the two ways to leave the transient state.
<!-- bilingual-en:end -->

> [!question]- 自检
> 标准有限吸收链中，$B$ 的每一行为什么应当加总为 1？
> <!-- bilingual-en:start -->
> In a standard finite absorbing chain, why must each row of $B$ sum to one?
> <!-- bilingual-en:end -->
>
> **答案：** 从每个暂态状态最终都会以概率一进入某个吸收状态；该行列出的正是这些互斥终点的概率。
> <!-- bilingual-en:start -->
> **Answer:** Eventual absorption occurs with probability one, and the row lists the probabilities of the mutually exclusive absorbing destinations.
> <!-- bilingual-en:end -->

## 来源与核验

- [Grinstead and Snell, Introduction to Probability, Theorem 11.6](https://math.dartmouth.edu/~prob/prob/prob.pdf#page=429)：核对 $B_{ia}$ 的概率解释、$B=NR$ 及证明。

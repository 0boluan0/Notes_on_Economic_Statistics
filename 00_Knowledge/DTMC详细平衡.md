---
aliases:
- Detailed balance
- DTMC detailed balance
- 详细平衡
student_os: knowledge-atom
atom_id: PROB-DTMC-020
atom_set: discrete-time-markov-chains
atom_type: definition
status: source-checked
requires:
- '[[Markov稳态分布]]'
related:
- '[[DTMC可逆性]]'
leads_to:
- '[[详细平衡与可逆]]'
part_of:
- '[[离散时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Detailed balance
<!-- bilingual-en:end -->

*DTMC 详细平衡*

<!-- bilingual-en:start -->
**Detailed balance** means that each pair of states exchanges equal amounts of probability in the two directions. Imagine many copies of a system: some move from working to failed, while others move from failed to working. The question concerns how much of the whole population moves each way, not just the chance that an individual copy moves.

Use a finite or countable [[离散时间Markov链|discrete-time Markov chain]] with a fixed transition rule. Here the [[DTMC时间齐次转移核|one-step probability]] $p_{ij}$ means “from $i$ to $j$”, so $P=(p_{ij})$ is [[Markov矩阵左右约定|row-stochastic]]: each row sums to one. A candidate [[概率质量函数|probability distribution]] $\pi$ assigns mass $\pi_i$ to state $i$, with $\pi_i\ge0$ and $\sum_i\pi_i=1$.
<!-- bilingual-en:end -->

**详细平衡**是说每一对状态在两个方向交换的概率量相等。想象同时运行很多个系统：一些由正常变成故障，另一些由故障恢复正常。要比较的是整个群体中各有多少比例向两边移动，而不只是某个系统移动的概率。

采用有限或可数状态、转移规则固定的[[离散时间Markov链]]。这里[[DTMC时间齐次转移核|一步转移概率]] $p_{ij}$ 指“从 $i$ 到 $j$”，所以 $P=(p_{ij})$ 是每行和为一的[[Markov矩阵左右约定|行随机矩阵]]。候选[[概率质量函数|概率分布]] $\pi$ 给状态 $i$ 分配质量 $\pi_i$，满足非负且总和为一。

<!-- bilingual-en:start -->
## Why a transition probability needs a weight
<!-- bilingual-en:end -->

*为什么还要乘上当前状态的概率*

<!-- bilingual-en:start -->
If the current state has distribution $\pi$, then

$$\Pr(X_n=i,X_{n+1}=j)=\pi_i p_{ij}.$$

This is the probability of first being at $i$ and then moving to $j$. For a state with $\pi_i=0$, its weighted outgoing flow is zero even if its transition row contains positive entries. We have not yet assumed that $\pi$ stays unchanged after the step; detailed balance will be a sufficient test for that.
<!-- bilingual-en:end -->

若当前状态服从 $\pi$，先处于 $i$、再移到 $j$ 的联合概率就是 $\pi_i p_{ij}$。某状态若 $\pi_i=0$，即使它的转移行有正条目，加权流出量仍为零。这里还没有假定 $\pi$ 经过一步后保持不变；详细平衡将给出一个充分的检查方法。

> [!important] Pairwise equality · 逐对相等
> <!-- bilingual-en:start -->
> $P$ satisfies **detailed balance relative to $\pi$** when
>
> $$\pi_i p_{ij}=\pi_jp_{ji}\qquad\text{for every }i,j.$$
>
> The case $i=j$ is automatic. The work lies in checking every distinct pair, including pairs with a zero transition probability in one direction.
> <!-- bilingual-en:end -->
>
> 对每个状态对都有上面的等式时，称 $P$ **相对于 $\pi$ 满足详细平衡**。$i=j$ 自动成立；实际需要核对所有不同状态对，包括某方向转移概率为零的配对。

> [!example] Unequal transition probabilities, equal flow · 概率不同，流量相等
> <!-- bilingual-en:start -->
> Let state 1 mean working and state 2 failed, with
>
> $$P=\begin{pmatrix}0.9&0.1\\0.4&0.6\end{pmatrix},\qquad\pi=(0.8,0.2).$$
>
> With 80% of the probability at state 1, $0.8\times0.1=0.08$ moves to state 2. The reverse flow is $0.2\times0.4=0.08$. Thus detailed balance holds even though $p_{12}=0.1$ and $p_{21}=0.4$ differ. Movement still occurs in both directions.
> <!-- bilingual-en:end -->
>
> 状态一表示正常、二表示故障。当前80%的概率在一，其中10%移到二，流量为 $0.08$；当前20%在二，其中40%移回一，流量也为 $0.08$。两边转移概率分别为 $0.1$ 和 $0.4$，并不相等；但加权后流量相等，而且两个方向仍都存在移动。

<!-- bilingual-en:start -->
For the general two-state matrix

$$P=\begin{pmatrix}1-a&a\\b&1-b\end{pmatrix},\qquad0<a,b\le1,$$

the nontrivial equation is $\pi_1a=\pi_2b$. Combining this with $\pi_1+\pi_2=1$ gives $\pi=(b/(a+b),a/(a+b))$. The numerical example uses $a=0.1,b=0.4$.
<!-- bilingual-en:end -->

一般二状态矩阵如上，其中 $0<a,b\le1$。唯一非平凡方程是 $\pi_1a=\pi_2b$，与总和一合起来得到 $\pi=(b/(a+b),a/(a+b))$。数值例子对应 $a=0.1,b=0.4$。

<!-- bilingual-en:start -->
## What the test does and does not establish
<!-- bilingual-en:end -->

*这项检查说明什么*

<!-- bilingual-en:start -->
A [[Markov稳态分布|stationary distribution]] only requires total incoming mass to restore each state's mass: in this row convention, $\pi P=\pi$. Detailed balance checks every pair separately, so it is stronger. A circulating current around three states can preserve each state's total mass while failing the pairwise test. The proof and counterexample are in [[详细平衡与可逆|the equivalence with reversibility]]. That theorem also connects these local equations to the whole-path meaning of [[DTMC可逆性|time reversibility]].

When using detailed balance to find $\pi$, solve the pair equations and still check nonnegativity and total mass one. On an infinite state space, nonnegative weights satisfying the equations may have infinite total mass and cannot then be normalized into a probability distribution. The definition here concerns a probability $\pi$, not an arbitrary invariant measure.
<!-- bilingual-en:end -->

[[Markov稳态分布|平稳分布]]只要求每个状态收到的总质量恢复其原质量，在本页行约定下写作 $\pi P=\pi$。详细平衡逐对检查，因此更强：三状态之间循环的概率流可以保持各处总量，却不通过逐对检查。证明和反例见[[详细平衡与可逆]]，它也把这些局部方程连接到整条路径的[[DTMC可逆性|时间可逆性]]。

用详细平衡求 $\pi$ 时，解出逐对方程后仍要检查非负性和总和一。无限状态空间中，满足方程的非负权重可能总量无穷，因而无法归一化成概率分布。本页定义采用概率 $\pi$，不是任意不变测度。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Cambridge, Markov Chains, §11.2, PDF p.46](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=46) — Pairwise balance, invariant measures and two-state examples.
- [Levin and Peres, Markov Chains and Mixing Times, §1.6, PDF pp.29–30](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf#page=29) — Equation (1.29), Proposition 1.20, and the distinction between balance and circulating flow.
<!-- bilingual-en:end -->

Cambridge 核对逐对平衡、不变测度与二状态例子。

Levin 与 Peres 核对式(1.29)、命题1.20，以及详细平衡与循环流的区别。

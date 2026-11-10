---
aliases:
  - CTMC detailed balance
  - Rate detailed balance
  - 速率详细平衡
student_os: knowledge-atom
atom_id: PROB-CTMC-019
atom_set: continuous-time-markov-chains
atom_type: definition
status: source-checked
requires:
  - "[[CTMC平稳分布]]"
related:
  - "[[DTMC详细平衡]]"
leads_to:
  - "[[CTMC详细平衡与可逆]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# CTMC detailed balance
<!-- bilingual-en:end -->

*CTMC 详细平衡*

<!-- bilingual-en:start -->
Imagine many copies of a system switching between states. Detailed balance says that, for every pair of states, the probability flowing one way is exactly matched by the probability flowing back. It compares **occupancy multiplied by speed**, so the two transition speeds need not be equal.
<!-- bilingual-en:end -->

想象有很多份相同系统在状态之间切换。详细平衡要求：任意一对状态之间，沿一个方向流动的概率，恰好被反方向的流动抵消。比较的是**占据概率乘以跳转速率**，所以两个方向的速率本身可以不同。

<!-- bilingual-en:start -->
In a [[CTMC时间齐次转移函数|time-homogeneous CTMC]], the [[生成矩阵约束|generator]] entry $q_{ij}$, for $i\ne j$, is the instantaneous rate of moving from $i$ to $j$. If a fraction $\pi_i$ of systems occupies $i$, then over a short interval $h$, the probability of an $i$-to-$j$ transition is $\pi_iq_{ij}h+o(h)$. The [[CTMC短时概率展开|short-time expansion]] explains the remainder $o(h)$. Thus $\pi_iq_{ij}$ measures probability flow per unit time.
<!-- bilingual-en:end -->

在[[CTMC时间齐次转移函数|时间齐次 CTMC]]中，[[生成矩阵约束|生成矩阵]]的非对角项 $q_{ij}$ 是从 $i$ 跳到 $j$ 的瞬时速率。若有比例 $\pi_i$ 的系统处在 $i$，那么短时间 $h$ 内，从 $i$ 到 $j$ 的转移概率为 $\pi_iq_{ij}h+o(h)$；余项的含义见[[CTMC短时概率展开]]。因此 $\pi_iq_{ij}$ 是每单位时间的概率流。

> [!important] Pairwise flow balance · 逐对流量平衡
> <!-- bilingual-en:start -->
> For a generator $Q$ and nonnegative weights $w_i$, not all zero, detailed balance means $w_iq_{ij}=w_jq_{ji}$ for every $i\ne j$. When the weights sum to one, write them as a probability vector $\pi$. Checking this equality does not require assuming beforehand that $\pi$ is stationary.
> <!-- bilingual-en:end -->
>
> 给定生成矩阵 $Q$ 和不全为零的非负权重 $w_i$，详细平衡要求每个 $i\ne j$ 都满足 $w_iq_{ij}=w_jq_{ji}$。权重之和为一时，可记为概率向量 $\pi$。检验这些等式时，不必预先假设 $\pi$ 已经平稳。

<!-- bilingual-en:start -->
## Different rates can balance
<!-- bilingual-en:end -->

## 不同速率也可以平衡

<!-- bilingual-en:start -->
Let state 0 mean “working” and state 1 mean “under repair”. A working device fails at rate 2 per hour; a broken device is repaired at rate 3 per hour:
<!-- bilingual-en:end -->

令状态0表示“工作”，状态1表示“维修”。工作中的设备以每小时2的速率故障，维修中的设备以每小时3的速率修复：

$$
Q=\begin{pmatrix}-2&2\\3&-3\end{pmatrix},
\qquad
\pi=(3/5,\,2/5).
$$

<!-- bilingual-en:start -->
The flow into repair is $(3/5)\times2=6/5$ per hour; the flow back to working is $(2/5)\times3=6/5$ per hour. Repair is faster, but fewer devices are being repaired. Those two effects compensate. In contrast, equal occupancy $(1/2,1/2)$ would produce flows 1 and $3/2$, which do not balance.
<!-- bilingual-en:end -->

进入维修的流量是 $(3/5)\times2=6/5$ 每小时；恢复工作的流量是 $(2/5)\times3=6/5$ 每小时。维修速度更快，但维修中的设备更少，两种效应正好抵消。若占据概率各为 $1/2$，两个流量分别是1和 $3/2$，就不平衡。

<!-- bilingual-en:start -->
## Weights first, probabilities second
<!-- bilingual-en:end -->

## 先求权重，再检查概率归一化

<!-- bilingual-en:start -->
Multiplying every weight by the same positive constant preserves detailed balance. In the device example, $w=(3,2)$ works before normalization. More generally, if $0<W=\sum_iw_i<\infty$, then $\pi_i=w_i/W$ is a probability vector. On infinitely many states, the sum may diverge: a chain on the integers with rate 1 to each neighbour satisfies detailed balance with $w_i=1$, but these particular weights cannot be normalized. This is why [[生灭链平稳递推|birth–death stationary recursions]] must check the total mass after finding relative weights.
<!-- bilingual-en:end -->

把全部权重乘以同一个正数，不会破坏详细平衡。设备例子可先取 $w=(3,2)$，再归一化。一般地，若 $0<W=\sum_iw_i<\infty$，则 $\pi_i=w_i/W$ 是概率向量。无限状态时，权重和可能发散：整数线上向左右邻居各以速率1跳转的链，满足权重 $w_i=1$ 的详细平衡，但这组权重不能归一化。这就是[[生灭链平稳递推]]求出相对权重后还要检查总质量的原因。

> [!tip] Local balance versus total balance · 逐对平衡与总量平衡
> <!-- bilingual-en:start -->
> For a finite chain, normalized detailed balance implies [[CTMC平稳分布|stationarity]] and [[CTMC可逆性|reversibility]]. Stationarity alone can allow a circulating flow around a loop. The [[CTMC详细平衡与可逆|equivalence theorem]] proves the implication and gives a three-state counterexample to its converse.
> <!-- bilingual-en:end -->
>
> 有限链中，归一化后的详细平衡会推出[[CTMC平稳分布|平稳性]]和[[CTMC可逆性|可逆性]]。只有平稳性时，仍然可能有绕着环流动的净概率流。[[CTMC详细平衡与可逆|等价定理]]解释这条推论，并用三状态反例说明平稳性为何不够。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Cambridge, Applied Probability, §2.6, PDF pp. 28–30](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf): stationary time reversal, rate detailed balance and their relationship.
- [Whitt, Continuous-Time Markov Chains (2012), §6, PDF pp. 33–36](https://www.columbia.edu/~ww2040/Whitt_CTMCnotes121312.pdf): reversed transition rates and detailed balance. The finite-state proofs and examples above also check the conclusions directly.
<!-- bilingual-en:end -->

- [Cambridge, Applied Probability，§2.6，PDF 第28–30页](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：平稳时间反演、速率详细平衡及两者的关系。
- [Whitt, Continuous-Time Markov Chains（2012），§6，PDF 第33–36页](https://www.columbia.edu/~ww2040/Whitt_CTMCnotes121312.pdf)：反演生成率与详细平衡；正文另以有限状态证明和具体算例直接核对。

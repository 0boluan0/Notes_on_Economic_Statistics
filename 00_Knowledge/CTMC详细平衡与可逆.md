---
aliases:
  - CTMC detailed balance and reversibility
  - 速率详细平衡与可逆等价定理
student_os: knowledge-atom
atom_id: PROB-CTMC-035
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
requires:
  - "[[CTMC平稳分布]]"
  - "[[有限CTMC平稳生成矩阵判据]]"
  - "[[CTMC详细平衡]]"
  - "[[CTMC可逆性]]"
leads_to:
  - "[[生灭链平稳递推]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# Detailed balance and reversibility for finite CTMCs
<!-- bilingual-en:end -->

*有限 CTMC 的详细平衡与可逆性*

<!-- bilingual-en:start -->
Checking an entire movie in reverse sounds harder than checking a few rates. For a finite time-homogeneous CTMC, these tasks are equivalent: pairwise balance of instantaneous probability flow guarantees that the whole equilibrium path has no preferred time direction.
<!-- bilingual-en:end -->

比较整段录像的正放和倒放，听起来比检查几个速率困难得多。但对有限状态时间齐次 CTMC，两件事是等价的：逐对瞬时概率流的平衡，能够保证整个平衡路径没有偏好的时间方向。

<!-- bilingual-en:start -->
Here [[CTMC详细平衡|detailed balance]] means $\pi_iq_{ij}=\pi_jq_{ji}$, whereas [[CTMC可逆性|reversibility]] is equality of all stationary finite-dimensional laws under time reflection. The [[CTMC平稳分布|stationary condition]] alone is $\pi P(t)=\pi$. Keep the same probability vector $\pi$ in all three statements.
<!-- bilingual-en:end -->

这里，[[CTMC详细平衡|详细平衡]]是 $\pi_iq_{ij}=\pi_jq_{ji}$；[[CTMC可逆性|可逆性]]是平稳路径的所有有限维分布在时间反射后不变；单独的[[CTMC平稳分布|平稳条件]]是 $\pi P(t)=\pi$。三种说法中必须使用同一个概率向量 $\pi$。

> [!important] Finite-state equivalence · 有限状态等价关系
> <!-- bilingual-en:start -->
> Let $Q$ be the generator of a finite time-homogeneous CTMC, and let $\pi_i\ge0$ with $\sum_i\pi_i=1$. Detailed balance holds if and only if the chain is reversible relative to $\pi$. Either condition implies stationarity. Irreducibility and strictly positive entries of $\pi$ are not required.
> <!-- bilingual-en:end -->
>
> 设 $Q$ 是有限状态时间齐次 CTMC 的生成矩阵，$\pi_i\ge0$ 且 $\sum_i\pi_i=1$。详细平衡当且仅当链相对于 $\pi$ 可逆；两者都推出平稳性。不要求链不可约，也不要求 $\pi$ 每项严格为正。

<!-- bilingual-en:start -->
## From local rates to a reversible path
<!-- bilingual-en:end -->

## 从局部速率到可逆路径

<!-- bilingual-en:start -->
First sum the pairwise balance equations over all states entering a fixed state $j$. The incoming flow equals $\pi_j\sum_{i\ne j}q_{ji}$, its outgoing flow. Since $q_{jj}=-\sum_{i\ne j}q_{ji}$, this gives $(\pi Q)_j=0$. The [[有限CTMC平稳生成矩阵判据|finite generator criterion]] then makes $\pi$ stationary.
<!-- bilingual-en:end -->

先固定状态 $j$，把流入它的逐对平衡等式相加。总流入等于 $\pi_j\sum_{i\ne j}q_{ji}$，也就是总流出。由于 $q_{jj}=-\sum_{i\ne j}q_{ji}$，得到 $(\pi Q)_j=0$。再由[[有限CTMC平稳生成矩阵判据|有限生成矩阵判据]]得到 $\pi$ 平稳。

<!-- bilingual-en:start -->
To propagate balance from an instant to an arbitrary time interval, write $n$ for the number of states and let $D=\operatorname{diag}(\pi_1,\ldots,\pi_n)$, the [[对角矩阵|diagonal matrix]] with these weights on its diagonal and zeros elsewhere; $Q^{\mathsf T}$ is the [[矩阵转置|transpose]], obtained by swapping rows and columns. Rate balance is the matrix equality $DQ=Q^{\mathsf T}D$. Repeated multiplication gives $DQ^k=(Q^{\mathsf T})^kD$. Substituting these identities into the [[CTMC矩阵指数|matrix-exponential series]] yields
<!-- bilingual-en:end -->

要把瞬间的平衡推广到任意长度的时间间隔，令 $n$ 表示状态数，$D=\operatorname{diag}(\pi_1,\ldots,\pi_n)$ 表示把这些权重放在对角线、其余位置填零的[[对角矩阵]]；$Q^{\mathsf T}$ 是交换行列得到的[[矩阵转置|转置]]。速率平衡写成 $DQ=Q^{\mathsf T}D$。反复相乘得到 $DQ^k=(Q^{\mathsf T})^kD$；代入[[CTMC矩阵指数|矩阵指数级数]]便有

$$
DP(t)=De^{tQ}=e^{tQ^{\mathsf T}}D=P(t)^{\mathsf T}D,
\qquad
\pi_ip_{ij}(t)=\pi_jp_{ji}(t).
$$

<!-- bilingual-en:start -->
Now consider successive observation gaps $h_r=t_r-t_{r-1}$. The [[CTMC定义|Markov property]] writes the joint probability as a starting weight times transition factors. Applying the last identity repeatedly reverses every factor:
<!-- bilingual-en:end -->

再看相邻观察间隔 $h_r=t_r-t_{r-1}$。[[CTMC定义|Markov 性]]把联合概率写成初始权重乘以各段转移概率；反复使用上一等式，就能逐个反转这些因子：

$$
\pi_{i_0}\prod_{r=1}^{m}p_{i_{r-1}i_r}(h_r)
=
\pi_{i_m}\prod_{r=1}^{m}p_{i_ri_{r-1}}(h_r).
$$

<!-- bilingual-en:start -->
The right side is the probability of the reflected observation: in chronological order it starts at $i_m$ and uses gaps $h_m,\ldots,h_1$. The matrix and repeated-balance arguments never divide by $\pi_i$, so they also cover zero weights. More concretely, if $\pi_i>0$ and $\pi_j=0$, detailed balance forces $q_{ij}=0$: the positive support is closed, and zero-weight states cannot be reached from it.
<!-- bilingual-en:end -->

右边正是时间反射后观察的概率：按时间先后，从 $i_m$ 开始，依次使用间隔 $h_m,\ldots,h_1$。矩阵证明和逐次平衡都没有除以 $\pi_i$，所以也覆盖零权重。具体地，若 $\pi_i>0$ 而 $\pi_j=0$，详细平衡迫使 $q_{ij}=0$：正概率支撑是闭的，从中无法到达零权重状态。

<!-- bilingual-en:start -->
## Why the converse holds
<!-- bilingual-en:end -->

## 为什么可逆性也推出详细平衡

<!-- bilingual-en:start -->
Reversibility includes two observation times, hence $\pi_ip_{ij}(h)=\pi_jp_{ji}(h)$. For $i\ne j$, divide by $h$ and let $h\downarrow0$. The [[CTMC短时概率展开|short-time rates]] give $\pi_iq_{ij}=\pi_jq_{ji}$. Thus comparing arbitrarily short reversed observations already recovers all the detailed-balance equations.
<!-- bilingual-en:end -->

可逆性包含两个观察时点的比较，因此 $\pi_ip_{ij}(h)=\pi_jp_{ji}(h)$。对 $i\ne j$，两边除以 $h$ 并令 $h\downarrow0$，由[[CTMC短时概率展开|短时速率展开]]得到 $\pi_iq_{ij}=\pi_jq_{ji}$。所以只要比较任意短的反向观察，就能恢复全部详细平衡方程。

<!-- bilingual-en:start -->
For a stationary finite chain, the reversed generator on states with $\pi_i>0$ can also be written $q^*_{ij}=\pi_jq_{ji}/\pi_i$ for $i\ne j$. [[Bayes法则|Bayes’ rule]] gives the weighting: reverse flow out of $i$ is forward flow into $i$, divided by its occupancy. Detailed balance is precisely $q^*_{ij}=q_{ij}$ on that support. The formula is not defined by division at zero-weight states.
<!-- bilingual-en:end -->

对平稳有限链，在 $\pi_i>0$ 的状态上，反演生成率也可写成 $q^*_{ij}=\pi_jq_{ji}/\pi_i$（$i\ne j$）。这是[[Bayes法则|Bayes 法则]]的加权：反向从 $i$ 流出的流量，是正向流入 $i$ 的流量，再除以占据概率。详细平衡恰好就是该支撑上的 $q^*_{ij}=q_{ij}$；不能在零权重状态上使用这个除法。

> [!example] Stationary circulation · 平稳环流
> <!-- bilingual-en:start -->
> Let the only allowed jumps be $1\to2\to3\to1$, each at rate 1. Its generator is $Q=\begin{pmatrix}-1&1&0\\0&-1&1\\1&0&-1\end{pmatrix}$. For $\pi=(1/3,1/3,1/3)$, every component of $\pi Q$ is zero: each state receives and sends flow $1/3$. Yet $\pi_1q_{12}=1/3$ while $\pi_2q_{21}=0$. The chain is stationary under $\pi$, but not reversible. Equal total inflow and outflow can coexist with a directed circulation.
> <!-- bilingual-en:end -->
>
> 只允许 $1\to2\to3\to1$，三条速率均为1。生成矩阵为 $Q=\begin{pmatrix}-1&1&0\\0&-1&1\\1&0&-1\end{pmatrix}$。取 $\pi=(1/3,1/3,1/3)$，$\pi Q$ 每项都是零：每个状态的总流入、总流出均为 $1/3$。但 $\pi_1q_{12}=1/3$，$\pi_2q_{21}=0$。所以从 $\pi$ 启动时链平稳，却不可逆。总流量守恒仍然允许定向环流。

<!-- bilingual-en:start -->
The finite-state hypothesis keeps all matrix sums and derivatives justified and determines $P(t)=e^{tQ}$ uniquely. On a countably infinite space, specify the process and its nonexplosion conditions before using analogous results; a formal equation $\pi Q=0$ is not a substitute for those checks. Also, a generator satisfying detailed balance does not make an arbitrary initial distribution stationary.
<!-- bilingual-en:end -->

有限状态条件保证这里的矩阵求和、求导都有依据，并使 $P(t)=e^{tQ}$ 唯一确定。推广到可数无穷空间前，要先明确过程及非爆炸条件；形式等式 $\pi Q=0$ 不能替代这些检查。另外，生成矩阵满足详细平衡，也不会让任意初始分布自动变成平稳分布。

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

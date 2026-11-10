---
aliases:
- Time-homogeneous DTMC
- Homogeneous DTMC transition kernel
- 离散时间齐次马尔可夫链
student_os: knowledge-atom
atom_id: PROB-DTMC-003
atom_set: discrete-time-markov-chains
atom_type: definition
status: source-checked
requires:
- '[[离散时间Markov链]]'
related:
- '[[Markov性]]'
- '[[Markov矩阵]]'
- '[[齐次不等于平稳]]'
- '[[CTMC时间齐次转移函数]]'
leads_to:
- '[[Markov矩阵左右约定]]'
- '[[DTMC转移复合]]'
part_of:
- '[[离散时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Time-homogeneous transition kernel
<!-- bilingual-en:end -->
*离散时间链的时间齐次转移核*

<!-- bilingual-en:start -->
A **time-homogeneous transition kernel** is one transition rule reused at every step. If a machine is working now, its chance of failing during the next hour is the same on Monday as on Friday, provided the model uses the same state description and one-hour step. The word “kernel” means a rule assigning a probability distribution for the next state to each current state.
<!-- bilingual-en:end -->

**时间齐次转移核**是一套在每一步重复使用的转移规则。若机器现在正常，那么在使用相同状态描述和一小时步长的模型里，周一下一小时发生故障的概率，与周五下一小时发生故障的概率相同。“核”指的是一种规则：对每个当前状态，都指定下一状态的概率分布。

<!-- bilingual-en:start -->
For a [[离散时间Markov链|discrete-time Markov chain]], let $p_n(i,j)$ denote the chance of moving from state $i$ to state $j$ at step $n$. The [[Markov性|Markov property]] already says that earlier states do not change that chance once the current state and the step are given. Homogeneity adds a new requirement: the rule does not change with $n$.
<!-- bilingual-en:end -->

对[[离散时间Markov链]]，用 $p_n(i,j)$ 表示第 $n$ 步从状态 $i$ 到状态 $j$ 的概率。[[Markov性]]已经要求：在给定当前状态与所处时点后，更早的状态不再改变这个概率。齐次性再增加一条要求：规则不随 $n$ 改变。

> [!abstract] One rule for every step · 每一步使用同一规则
> <!-- bilingual-en:start -->
> On a finite or countable state space $S$, a homogeneous kernel consists of numbers $p_{ij}$ satisfying
> <!-- bilingual-en:end -->
>
> 在有限或可数状态空间 $S$ 上，齐次核由满足下式的一组 $p_{ij}$ 组成：
>
> $$p_{ij}\ge0,\qquad \sum_{j\in S}p_{ij}=1,\qquad
> \Pr(X_{n+1}=j\mid X_n=i)=p_{ij}\quad\text{for all }n.$$
>
> <!-- bilingual-en:start -->
> The conditional-probability equality is read when $\Pr(X_n=i)>0$. The model specifies the rule at every state even when that state currently has zero probability. The observed law from one initial distribution need not identify rules at unreachable states, so arbitrary differences only there do not by themselves disprove the existence of a homogeneous representation of that law.
> <!-- bilingual-en:end -->
>
> 其中条件概率等式在 $\Pr(X_n=i)>0$ 时按普通条件概率理解。模型仍会对每个状态指定规则，即使它目前的概率为零。从一个初始分布得到的观测律，未必能确定不可达状态的规则；因此，若差异只发生在那里，不能仅凭那些任意取值就否定该观测律存在齐次表示。

<!-- bilingual-en:start -->
In the [[Markov矩阵左右约定|row-stochastic convention]], the finite-state kernel is the [[Markov矩阵|Markov matrix]] $P=(p_{ij})$: row $i$ lists the possible destinations from $i$. For a countably infinite state space the same entries form an infinite transition array; finite-dimensional matrix spectral theorems do not automatically apply to it.
<!-- bilingual-en:end -->

采用[[Markov矩阵左右约定|行随机约定]]，有限状态核就是 [[Markov矩阵]] $P=(p_{ij})$：第 $i$ 行列出从 $i$ 出发的各个去向。若状态可数无穷，这些数构成无限转移数组，不能自动套用有限维矩阵的谱定理。

<!-- bilingual-en:start -->
## The same number of steps gives the same transition
<!-- bilingual-en:end -->
*相同的步数间隔，对应相同的转移*

<!-- bilingual-en:start -->
Starting from state $i$, two steps use the same rule twice, regardless of the calendar date. Sum over the intermediate state $k$ to obtain $p_{ij}^{(2)}=\sum_kp_{ik}p_{kj}$. Iterating this [[DTMC转移复合|transition composition]] gives
<!-- bilingual-en:end -->

从状态 $i$ 出发，两步就是把同一规则连续用两次，与日历起点无关。对中间状态 $k$ 求和，得到 $p_{ij}^{(2)}=\sum_kp_{ik}p_{kj}$。反复应用这种[[DTMC转移复合]]，得到

$$\Pr(X_{n+m}=j\mid X_n=i)=p_{ij}^{(m)}=(P^m)_{ij},\qquad m\ge0.$$

<!-- bilingual-en:start -->
Here the superscript $(m)$ means **number of steps**, not time $n$; $P^0=I$ means that zero steps leave the state unchanged. The conditional form again requires positive probability of the conditioning state, while the transition rule itself specifies every starting state. For a nonhomogeneous row-kernel model, the corresponding product is $P_nP_{n+1}\cdots P_{n+m-1}$, so it can depend on both the starting time and the number of steps.
<!-- bilingual-en:end -->

这里上标 $(m)$ 表示**经过几步**，不是起始时点 $n$；$P^0=I$ 表示零步后状态不变。条件概率写法仍要求条件状态具有正概率，转移规则本身则规定每个出发状态。对非齐次行随机核，相应乘积是 $P_nP_{n+1}\cdots P_{n+m-1}$，因而可能同时依赖起始时间与步数。

> [!example] A fixed machine rule · 固定的设备规则
> <!-- bilingual-en:start -->
> Order the states as working, failed, and let
> <!-- bilingual-en:end -->
>
> 按正常、故障排列状态，设
>
> $$P=\begin{pmatrix}0.9&0.1\\0.4&0.6\end{pmatrix}.$$
>
> <!-- bilingual-en:start -->
> A working machine fails next hour with probability $0.1$. Two hours later it is failed either by staying working then failing, or by failing then remaining failed. These disjoint paths give $0.9\times0.1+0.1\times0.6=0.15$. The answer is the same from any starting hour at which the machine is working.
> <!-- bilingual-en:end -->
>
> 正常设备下一小时故障的概率为 $0.1$。两小时后故障有两条互斥路径：先正常再故障，或先故障并继续故障，所以概率为 $0.9\times0.1+0.1\times0.6=0.15$。从哪个小时开始，只要起点状态为正常，答案都相同。

<!-- bilingual-en:start -->
If a known hot season changes the working-to-failed probability from $0.1$ to $0.2$, the rule depends on the calendar and this model is not homogeneous. It may still be Markov: after the current state and time are known, the earlier path may add no predictive information. A weekday/weekend rule has the same distinction. If unrecorded age also matters beyond current state and time, the Markov property itself needs reconsideration.
<!-- bilingual-en:end -->

若已知高温季节把正常到故障的概率从 $0.1$ 改为 $0.2$，规则就依赖日历，该模型不是齐次的。它仍然可能具有 Markov 性：给定当前状态与时点后，更早路径仍可能不再补充预测信息。工作日与周末使用不同概率，也是同一区分。若未记录的年龄在当前状态和时点之外还起作用，则 Markov 性本身也需要重新检查。

> [!warning] A fixed rule does not freeze the distribution · 规则固定，分布仍可变化
> <!-- bilingual-en:start -->
> Starting the fixed-matrix example with certainty in the working state gives probability rows $(1,0)$, then $(0.9,0.1)$, then $(0.85,0.15)$. The matrix stayed fixed while the state distribution changed. [[齐次不等于平稳|Homogeneity and stationarity]] answer different questions; a [[Markov稳态分布|stationary starting distribution]] would instead remain fixed under the same rule.
> <!-- bilingual-en:end -->
>
> 上述固定矩阵若从确定正常开始，概率行依次为 $(1,0)$、$(0.9,0.1)$、$(0.85,0.15)$。矩阵没有变化，状态分布仍在变化。[[齐次不等于平稳|齐次与平稳]]回答不同问题；若从[[Markov稳态分布|平稳分布]]开始，分布才会在同一规则下保持不变。

## 来源与核验

<!-- bilingual-en:start -->
- [Gallager, MIT 6.262 Chapter 3, PDF pp. 1–3, (3.1)–(3.4)](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/3558b08622765d26c2b0a7d2eeeac885_MIT6_262S11_chap03.pdf): supports fixed versus time-dependent transition rules, initial laws and probability propagation. The machine examples apply these rules directly.
- [Weber, Cambridge Markov Chains, §1.5, PDF p. 7 / printed p. 3](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=7): supports multistep probabilities, row-law evolution and the matrix-power rule. The nonhomogeneous product uses the same successive conditioning with the appropriate matrix at each time.
- [Pitman, Berkeley Stat205B Lecture 1, Definitions 1.2–1.3](https://www.stat.berkeley.edu/~pitman/s205s03/lecture1.pdf): supports a kernel as a distribution-valued transition rule and the removal of its time index in the homogeneous case.
<!-- bilingual-en:end -->

- Gallager 第 1–3 页核对固定核、非齐次核、初始律与概率传播；设备例子按这些规则直接计算。Weber 第 1.5 节（PDF 第 7 页）核对多步转移和矩阵幂，非齐次乘积按各时点相应规则逐步复合。Pitman 定义 1.2–1.3 核对核的含义及齐次时不再带时间下标。

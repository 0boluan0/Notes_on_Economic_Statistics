---
aliases:
- Reversible Markov chain
- DTMC reversibility
- 时间可逆 Markov 链
student_os: knowledge-atom
atom_id: PROB-DTMC-031
atom_set: discrete-time-markov-chains
atom_type: definition
status: source-checked
requires:
- '[[离散时间Markov链]]'
- '[[Markov稳态分布]]'
related:
- '[[DTMC详细平衡]]'
leads_to:
- '[[详细平衡与可逆]]'
part_of:
- '[[离散时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Time reversibility
<!-- bilingual-en:end -->

*DTMC 时间可逆性*

<!-- bilingual-en:start -->
A **time-reversible Markov chain** produces stationary recordings whose probability law is unchanged when they are played backwards. This does not mean that one realised recording must look identical in both directions. It means that every possible sequence and its reverse have equal probabilities.

Consider a finite or countable [[离散时间Markov链|discrete-time Markov chain]] with fixed row transition matrix $P$. Start it in a [[Markov稳态分布|stationary probability distribution]] $\pi$: $X_0\sim\pi$ and $\pi P=\pi$. Consequently every $X_n$ has distribution $\pi$. The next question concerns their [[有限维分布|joint distribution]], which records how states at several times occur together.
<!-- bilingual-en:end -->

**时间可逆的 Markov 链**在平稳状态下录制的状态序列，倒放后仍有相同的概率规律。它不要求某一次实际录像正反看起来完全一样，而是要求每条可能序列与其倒序具有相同概率。

考虑有限或可数状态、行转移矩阵 $P$ 固定的[[离散时间Markov链]]。从[[Markov稳态分布|平稳概率分布]] $\pi$ 启动，即 $X_0\sim\pi$ 且 $\pi P=\pi$，于是每个 $X_n$ 都服从 $\pi$。进一步要考察的是多个时点共同取值的[[有限维分布|联合分布]]。

> [!important] A symmetry of joint laws · 联合律的对称性
> <!-- bilingual-en:start -->
> The chain is **reversible relative to $\pi$** if, for every integer $m\ge1$,
>
> $$(X_0,X_1,\ldots,X_m)\overset d=(X_m,X_{m-1},\ldots,X_0).$$
>
> The symbol $\overset d=$ means equality in distribution. Equivalently, for every state sequence $i_0,\ldots,i_m$, its forward probability equals the probability of $i_m,\ldots,i_0$ under the same stationary chain. The right side reverses the order of the observations; it does not replace $P$ by its matrix inverse.
> <!-- bilingual-en:end -->
>
> 每个整数 $m\ge1$ 都满足上述同分布关系，就称链**相对于 $\pi$ 可逆**。符号 $\overset d=$ 表示同分布；等价地，每条状态序列和它的倒序，在同一条平稳链下具有相同概率。右边是把观测顺序倒过来，并不是求 $P$ 的逆矩阵。

<!-- bilingual-en:start -->
## Read one path probability
<!-- bilingual-en:end -->

*怎样读出一段路径的概率*

<!-- bilingual-en:start -->
The [[Markov性|Markov property]] gives

$$\Pr_\pi(X_0=i_0,\ldots,X_m=i_m)
=\pi_{i_0}p_{i_0i_1}\cdots p_{i_{m-1}i_m}.$$

The subscript $\pi$ says that the initial law is $\pi$. We multiply its starting mass by each transition probability in order. To test reversal, compare this expression with $\pi_{i_m}p_{i_mi_{m-1}}\cdots p_{i_1i_0}$. The criterion [[详细平衡与可逆|detailed balance and reversibility]] proves that checking all two-state flows is enough to establish equality for every length; this shortcut uses the Markov structure.
<!-- bilingual-en:end -->

[[Markov性]]给出上面的路径乘积：下标 $\pi$ 说明初始律为 $\pi$，先取起始状态的概率，再依次乘各步转移概率。检验倒放时，把它与 $\pi_{i_m}p_{i_mi_{m-1}}\cdots p_{i_1i_0}$ 比较。[[详细平衡与可逆]]证明，只要核对全部两状态流量，就能保证任意长度的等式；这条捷径依赖 Markov 结构。

> [!example] Alternation can be reversible · 交替运动也可以可逆
> <!-- bilingual-en:start -->
> For $P=\begin{pmatrix}0&1\\1&0\end{pmatrix}$ and $\pi=(1/2,1/2)$, a stationary recording alternates between states. At any fixed length, there are two possible alternating sequences, each of probability $1/2$. Reversal preserves these sequences or swaps them, so it preserves their probabilities. Every nonalternating sequence and its reverse have probability zero.
>
> This chain has [[状态周期|period]] two. Reversibility therefore does not require aperiodicity or convergence from every initial state.
> <!-- bilingual-en:end -->
>
> 对上述交替矩阵及均匀分布，固定长度的平稳录像只有两条可能的交替序列，各有概率 $1/2$。倒放要么保持它们，要么交换它们，概率不变；不交替的序列及其倒序都不可能发生。该链的[[状态周期]]为二，因此可逆性不要求非周期，也不要求任意初态都逐步收敛。

<!-- bilingual-en:start -->
## Keep stationarity and the initial law in view
<!-- bilingual-en:end -->

*同时留意平稳性与实际初始律*

<!-- bilingual-en:start -->
Stationarity means that shifting all observation times together preserves their joint law. Reversibility means that reversing their order preserves it. For a homogeneous Markov chain, starting in an invariant distribution gives the first property; it does not automatically give the second. A uniform distribution circulating around a directed three-state ring is stationary, yet the pair $(1,2)$ can occur and $(2,1)$ cannot. The full calculation is in [[详细平衡与可逆|the stationarity counterexample]].

The initial law is essential. If the alternating chain above starts at state 1 with certainty, $(X_0,X_1)=(1,2)$ with probability one. Its reversed pair is $(2,1)$, so this particular nonstationary experiment is not invariant under reversal, even though the matrix is reversible relative to the uniform law. See [[齐次不等于平稳|homogeneity versus stationarity]] for why a fixed transition rule does not choose the initial law.

No irreducibility assumption is needed for this definition. If $\pi$ gives some states zero mass, those states are never seen in the stationary recording. Reversibility relative to $\pi$ describes that stationary experiment; it need not describe experiments started in those other states.
<!-- bilingual-en:end -->

平稳性说的是把全部观测时点一起平移，联合律不变；可逆性说的是把观测次序反过来，联合律不变。齐次 Markov 链从不变分布开始就具有前者，但未必具有后者。均匀分布沿三状态单向环循环时仍平稳，可是配对 $(1,2)$ 能发生，$(2,1)$ 不能；完整计算见[[详细平衡与可逆|平稳但不可逆的反例]]。

实际初始律很重要。上述交替链若确定从状态一开始，$(X_0,X_1)$ 必为 $(1,2)$，而倒序配对为 $(2,1)$，这次非平稳实验便不具有倒放不变性，尽管该矩阵相对于均匀律可逆。[[齐次不等于平稳]]解释了为什么固定转移规则没有替你选定初始律。

本定义不要求不可约。若 $\pi$ 给某些状态零质量，它们不会出现在平稳录像里；相对于 $\pi$ 的可逆性描述的是这次平稳实验，不必适用于从那些其他状态启动的实验。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Cambridge, Markov Chains, §§11.1–11.3, PDF pp.45–47](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=45) — Stationary time reversal and path-law invariance; its irreducible formulation is distinguished from the general definition used here.
- [Levin and Peres, Markov Chains and Mixing Times, §1.6, PDF pp.29–30](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf#page=29) — Equations (1.30)–(1.31) for finite paths, and the distinction between a chain and its time reversal.
<!-- bilingual-en:end -->

Cambridge 核对平稳时间反演与路径律不变性；原源的不可约表述与本页的一般定义作区分。

Levin 与 Peres 核对有限路径式(1.30)–(1.31)，以及原链与时间反演链的区别。

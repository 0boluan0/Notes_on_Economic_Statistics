---
aliases:
  - Doob decomposition
  - Doob decomposition theorem
  - 离散Doob分解
student_os: knowledge-atom
atom_id: PROB-MG-007
atom_set: martingales-stopping
atom_type: theorem
status: source-checked
requires:
  - "[[次鞅]]"
  - "[[离散可预测过程]]"
related:
  - "[[鞅定义]]"
  - "[[鞅差序列]]"
  - "[[鞅差部分和]]"
part_of:
  - "[[鞅与停时.canvas]]"
---

<!-- bilingual-en:start -->
# Discrete Doob decomposition
<!-- bilingual-en:end -->

*离散 Doob 分解*

<!-- bilingual-en:start -->
A process can move because of an expected gain already visible from the past, and because of a new change whose conditional mean is zero. **Doob decomposition separates these two contributions.** For a discrete-time submartingale, the accumulated expected gains form a predictable, nondecreasing process; subtracting them leaves a martingale. “Predictable” refers to information, not to a deterministic trend: tomorrow's expected gain may depend on today's random outcome.
<!-- bilingual-en:end -->

一个过程的变化可以包含两部分：根据过去已经能预期的收益，以及条件均值为零的新变化。**Doob 分解把这两种贡献拆开。** 对离散时间次鞅，把预期收益逐步累加，会得到可预测、非递减的过程；减去它之后，剩下的就是鞅。“可预测”描述信息，不要求趋势是确定的：明天的预期收益可以取决于今天的随机结果。

<!-- bilingual-en:start -->
For a small example, toss two independent fair coins. A head contributes $+1$ and a tail $-1$; denote these changes by $\xi_1,\xi_2$. Start at $X_0=0$. After the first toss set $X_1=\xi_1$. Before the second toss, pay a bonus of $1$ if the first toss was a head, then add the second fair change. With $\mathbf1_A$ denoting the [[事件指示变量|indicator]] of event $A$, this is
<!-- bilingual-en:end -->

先看两个独立公平硬币的例子。正面贡献 $+1$，反面贡献 $-1$，分别记为 $\xi_1,\xi_2$。从 $X_0=0$ 开始，第一次后令 $X_1=\xi_1$。第二次抛掷前，如果第一次为正面就支付 $1$ 的奖励，然后再加上第二次的公平变化。用 $\mathbf1_A$ 表示事件 $A$ 的[[事件指示变量|指示变量]]，即

$$X_2=\xi_1+\mathbf1_{\{\xi_1=1\}}+\xi_2.$$

<!-- bilingual-en:start -->
At time $1$, the bonus is already known, but the second toss is not. Its conditional expected contribution is zero. Thus the predictable part at time $2$ is $A_2=\mathbf1_{\{\xi_1=1\}}$, while the martingale part is $M_2=\xi_1+\xi_2$. The four equally likely paths show both parts:
<!-- bilingual-en:end -->

到时刻 $1$，奖励是否支付已经知道，第二次抛掷却还未知，而它的条件期望贡献为零。因此，时刻 $2$ 的可预测部分为 $A_2=\mathbf1_{\{\xi_1=1\}}$，鞅部分为 $M_2=\xi_1+\xi_2$。四条等可能路径如下：

| $\xi_1$ | $\xi_2$ | $X_1$ | $X_2$ | $A_2$ | $M_2=X_2-A_2$ |
|---|---|---|---|---|---|
| $-1$ | $-1$ | $-1$ | $-2$ | $0$ | $-2$ |
| $-1$ | $1$ | $-1$ | $0$ | $0$ | $0$ |
| $1$ | $-1$ | $1$ | $1$ | $1$ | $0$ |
| $1$ | $1$ | $1$ | $3$ | $1$ | $2$ |

<!-- bilingual-en:start -->
Here $A_0=A_1=0$ and $M_0=0,M_1=\xi_1$. After a first tail, the next values $-2,0$ average to $X_1=-1$; after a first head, the next values $1,3$ average to $X_1+1=2$. This verifies the submartingale inequality. The first path also shows why an increasing predictable part does not make the original path increasing.
<!-- bilingual-en:end -->

这里 $A_0=A_1=0$，$M_0=0,M_1=\xi_1$。第一次为反面时，下一步的 $-2,0$ 平均为 $X_1=-1$；第一次为正面时，下一步的 $1,3$ 平均为 $X_1+1=2$，因此满足次鞅不等式。第一条路径也说明：可预测部分递增，不意味着原过程的每条路径递增。

> [!important] The discrete-time theorem · 离散时间定理
> <!-- bilingual-en:start -->
> Fix a [[滤过定义|filtration]] $(\mathcal F_n)_{n\ge0}$, the information available by each integer time. Let $X_n$ be a real [[次鞅|submartingale]] for this filtration: it is [[适应过程|adapted]], satisfies $E|X_n|<\infty$ at every finite $n$, and $E[X_n\mid\mathcal F_{n-1}]\ge X_{n-1}$ a.s. for $n\ge1$. Then $X_n=M_n+A_n$, where $M$ is a [[鞅定义|martingale]] and $A$ is integrable, $A_0=0$, and $A_n\ge A_{n-1}$ a.s. Also, $A_n$ is $\mathcal F_{n-1}$-measurable for $n\ge1$: it is a [[离散可预测过程|predictable process]]. With $A_0=0$, these two processes are unique up to almost-sure equality at every time. No finite second moment is required.
> <!-- bilingual-en:end -->
>
> 固定[[滤过定义|滤过]] $(\mathcal F_n)_{n\ge0}$，记录每个整数时刻已知的信息。设实值 $X_n$ 是该滤过下的[[次鞅]]：它[[适应过程|适应]]，每个有限 $n$ 都满足 $E|X_n|<\infty$，且 $n\ge1$ 时 $E[X_n\mid\mathcal F_{n-1}]\ge X_{n-1}$ 几乎必然成立。则可写成 $X_n=M_n+A_n$，其中 $M$ 是[[鞅定义|鞅]]，$A$ 可积、$A_0=0$，并且 $A_n\ge A_{n-1}$ 几乎必然成立。此外，$n\ge1$ 时 $A_n$ 对 $\mathcal F_{n-1}$ 可测，即它是[[离散可预测过程]]。固定 $A_0=0$ 后，两个过程在每一时刻几乎必然相等的意义下唯一。这里不要求有限二阶矩。

<!-- bilingual-en:start -->
## Accumulate the conditional expected changes
<!-- bilingual-en:end -->

## 分解的构造

<!-- bilingual-en:start -->
At step $k$, compare the [[条件期望|conditional prediction]] of $X_k$ using the previous information with the value $X_{k-1}$ already known. Call the difference $d_k$. It is the conditional expected change, not the change actually realized. Define
<!-- bilingual-en:end -->

在第 $k$ 步，比较使用上期信息对 $X_k$ 作出的[[条件期望|条件预测]]与已经知道的 $X_{k-1}$，把差记为 $d_k$。它是条件期望增量，不是实际实现的增量。定义

$$
d_k=E[X_k\mid\mathcal F_{k-1}]-X_{k-1},\qquad
A_0=0,\qquad A_n=\sum_{k=1}^{n}d_k,\qquad M_n=X_n-A_n.
$$

<!-- bilingual-en:start -->
Each $d_k$ is nonnegative by the submartingale property and is known at time $k-1$. For $k\le n$, that information is contained in $\mathcal F_{n-1}$, so the finite sum $A_n$ is also known one step before $n$. This proves predictability and monotonicity. The [[期望|integrability]] check is equally finite:
<!-- bilingual-en:end -->

每个 $d_k$ 由次鞅性质非负，并且在 $k-1$ 时已经知道。若 $k\le n$，这些信息都包含在 $\mathcal F_{n-1}$ 中，所以有限和 $A_n$ 在 $n$ 前一步也已经知道。这就证明了可预测性和单调性。[[期望|可积性]]也只需检查有限项：

$$
E|d_k|\le E|X_k|+E|X_{k-1}|<\infty,
\qquad E|A_n|\le\sum_{k=1}^{n}E|d_k|<\infty.
$$

<!-- bilingual-en:start -->
The first bound uses $|E[X_k\mid\mathcal F_{k-1}]|\le E[|X_k|\mid\mathcal F_{k-1}]$ and [[全期望公式|total expectation]]. Therefore $M_n=X_n-A_n$ is integrable and adapted. Since $A_n$ is already known at time $n-1$, conditioning gives
<!-- bilingual-en:end -->

第一个估计使用 $|E[X_k\mid\mathcal F_{k-1}]|\le E[|X_k|\mid\mathcal F_{k-1}]$ 和[[全期望公式]]。因此 $M_n=X_n-A_n$ 可积且适应。$A_n$ 在 $n-1$ 时已经知道，条件化后有

$$
E[M_n\mid\mathcal F_{n-1}]
=E[X_n\mid\mathcal F_{n-1}]-A_n
=X_{n-1}-A_{n-1}=M_{n-1}.
$$

<!-- bilingual-en:start -->
So $M$ is a martingale. Equivalently, its increments are $M_n-M_{n-1}=X_n-E[X_n\mid\mathcal F_{n-1}]$, a [[鞅差序列|martingale difference sequence]]: the observed value minus its previous prediction. The proof works for a finite horizon $0,\ldots,N$, or for all nonnegative integer times with integrability checked separately at each time. It makes no claim about convergence as $n\to\infty$.
<!-- bilingual-en:end -->

因此 $M$ 是鞅。等价地，增量为 $M_n-M_{n-1}=X_n-E[X_n\mid\mathcal F_{n-1}]$，构成[[鞅差序列]]：实现值减去此前预测。证明既适用于有限期限 $0,\ldots,N$，也适用于所有非负整数时刻，只需逐时检查可积性。它没有断言 $n\to\infty$ 时的收敛。

<!-- bilingual-en:start -->
## Why the initial value matters
<!-- bilingual-en:end -->

## 规范化保证唯一性

<!-- bilingual-en:start -->
Suppose another decomposition $X_n=\widetilde M_n+\widetilde A_n$ has the same properties, including $\widetilde A_0=0$. The martingale equality and predictability force
<!-- bilingual-en:end -->

假设另一个分解 $X_n=\widetilde M_n+\widetilde A_n$ 满足相同性质，包括 $\widetilde A_0=0$。鞅等式与可预测性迫使

$$
\widetilde A_n-\widetilde A_{n-1}
=E[X_n\mid\mathcal F_{n-1}]-X_{n-1}=d_n.
$$

<!-- bilingual-en:start -->
Starting from zero, these forced increments determine every $\widetilde A_n=A_n$, and then $\widetilde M_n=M_n$. Without the initial normalization, even a deterministic constant $c$ could be moved between the parts: $X_n=(M_n+c)+(A_n-c)$. Both properties would remain, so uniqueness would fail. Because discrete time is countable, the almost-sure equalities can be made simultaneous for all $n$ outside one null event.
<!-- bilingual-en:end -->

从零出发，这些被强制确定的增量决定了每个 $\widetilde A_n=A_n$，进而 $\widetilde M_n=M_n$。若不固定初值，哪怕一个确定常数 $c$ 也可以在两部分之间移动：$X_n=(M_n+c)+(A_n-c)$，两类性质仍然成立，唯一性便失败。离散时刻可数，因此还可以在排除一个零概率事件后，让所有 $n$ 的等式同时成立。

<!-- bilingual-en:start -->
## Interpret the pieces relative to the information
<!-- bilingual-en:end -->

## 分解在区分什么

<!-- bilingual-en:start -->
In the coin example, the bonus is random before the first toss and known afterwards. Its contribution belongs in $A$ because it is predictable when the second step is taken. The remaining fair change belongs in $M$. If $X$ is already a martingale, every $d_k=0$, so $A_n=0$ and $M_n=X_n$.
<!-- bilingual-en:end -->

在硬币例中，奖励在第一次抛掷前是随机的，第一次之后就已知。它在第二步发生时可预测，因此归入 $A$；剩下的公平变化归入 $M$。若 $X$ 本来就是鞅，则每个 $d_k=0$，于是 $A_n=0$、$M_n=X_n$。

> [!note] Prediction is tied to a chosen filtration · 预测依赖选定的信息
> <!-- bilingual-en:start -->
> The decomposition describes conditional means, not a causal explanation of the process. Changing the information can change $E[X_k\mid\mathcal F_{k-1}]$ and may even destroy the submartingale property. Apply the theorem only after specifying the filtration and checking its hypotheses. This is the discrete decomposition; a continuous-time decomposition requires a separate theorem and additional conditions.
> <!-- bilingual-en:end -->
>
> 分解描述条件均值，不是对过程作因果解释。更换信息可能改变 $E[X_k\mid\mathcal F_{k-1}]$，甚至使次鞅性质失效。必须先指定滤过并检查假设，再应用定理。这里是离散分解；连续时间分解需要另外的定理与条件。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 15.070J, Lecture 10, PDF pp.2–3](https://ocw.mit.edu/courses/15-070j-advanced-stochastic-processes-fall-2013/36fffa59f09fbedd5969539c90df4cbb_MIT15_070JF13_Lec10.pdf#page=2): discrete predictability, the conditional-increment construction, integrability and the martingale check. The normalization used in the proof is stated explicitly here.
- [Durrett, Probability: Theory and Examples, fifth-edition draft, Theorem 4.3.2, printed p.225 / PDF p.233](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf#page=233): the normalized decomposition and its uniqueness. The sum above uses the step index consistently with the recurrence (4.3.1); the finite coin example is directly computed here.
<!-- bilingual-en:end -->

- MIT 15.070J 第10讲，PDF2–3页：离散可预测性、条件增量构造、可积性与鞅性质验证；本文在定理中明确写出原证明使用的初值规范化。
- Durrett《Probability: Theory and Examples》第五版草稿，定理4.3.2，印刷225页／PDF233页：从零开始的分解与唯一性；本文求和的逐步指标与递推式(4.3.1)保持一致，并直接计算有限硬币例。

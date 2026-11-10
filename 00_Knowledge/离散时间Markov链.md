---
aliases:
- Discrete-time Markov chain
- DTMC
- 离散时间马尔可夫链
student_os: knowledge-atom
atom_id: PROB-DTMC-024
atom_set: discrete-time-markov-chains
atom_type: definition
status: source-checked
requires:
- '[[随机过程]]'
- '[[Markov性]]'
related:
- '[[Markov矩阵]]'
leads_to:
- '[[Markov充分状态]]'
- '[[DTMC时间齐次转移核]]'
part_of:
- '[[离散时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Discrete-time Markov chain
<!-- bilingual-en:end -->
*离散时间 Markov 链*

<!-- bilingual-en:start -->
A **discrete-time Markov chain** describes a system observed in steps, where knowing its current state is enough to predict the next state at that step. For example, we can model a machine as “working” or “failed” at the start of each hour. A chain specifies probabilities for what happens next; it does not prescribe one certain sequence of states.
<!-- bilingual-en:end -->

**离散时间 Markov 链**描述按步骤观察的系统：在某一步，知道当前状态就足以预测下一状态。例如，可以记录一台设备在每小时开始时是“正常”还是“故障”。链给出的是接下来各种变化的概率，不是一条事先确定的状态序列。

<!-- bilingual-en:start -->
Write the observations as $X_0,X_1,X_2,\ldots$, a [[随机过程|stochastic process]] indexed by nonnegative integers. $X_n$ is the state at step $n$, and the **state space** $S$ lists its allowed values. In this course's finite/countable-state convention, $S$ can be a finite list, such as two weather states, or a countably infinite list, such as the nonnegative integers counting customers in a queue. A time step can represent an hour, a day or one completed update; its unit must be chosen for the model.
<!-- bilingual-en:end -->

把这些观测记为 $X_0,X_1,X_2,\ldots$，就是一个用非负整数编号的[[随机过程]]。$X_n$ 是第 $n$ 步的状态，**状态空间** $S$ 列出允许取的值。本课程采用有限或可数状态的口径：$S$ 可以只有晴、雨两个值，也可以是排队人数所用的非负整数。一步可以是一小时、一天或一次完成的更新，具体单位由模型规定。

> [!abstract] What makes it a Markov chain · 什么使它成为 Markov 链
> <!-- bilingual-en:start -->
> The extra requirement is the [[Markov性|Markov property]]: for every history with positive probability and each next state $j$,
> <!-- bilingual-en:end -->
>
> 额外要求是 [[Markov性]]：对每段具有正概率的历史与任意下一状态 $j$，
>
> $$\Pr(X_{n+1}=j\mid X_0=i_0,\ldots,X_n=i)
> =\Pr(X_{n+1}=j\mid X_n=i).$$
>
> <!-- bilingual-en:start -->
> Knowing the earlier path must not change the next-state probability after the present is known. Merely recording a system at regular times does not establish this equality. For instance, a “working” label may omit the machine's age when age changes its failure risk; a more informative [[Markov充分状态|state description]] may be needed.
> <!-- bilingual-en:end -->
>
> 知道当前状态以后，更早路径不能再改变下一状态的概率。定时记录一个系统，并不能证明这个等式。例如，若机器年龄会改变故障风险，只记“正常”可能遗漏年龄，因而需要更充分的[[Markov充分状态|状态描述]]。

<!-- bilingual-en:start -->
## A starting law and a transition rule
<!-- bilingual-en:end -->
*从哪里开始，下一步如何变化*

<!-- bilingual-en:start -->
To specify a chain, give its **initial distribution** $\mu_0(i)=\Pr(X_0=i)$ and its transition rules $p_n(i,j)$ from state $i$ to state $j$ at step $n$. Each row must have nonnegative entries summing to one. With the [[Markov矩阵左右约定|row-stochastic convention]], these entries form $P_n$, and a probability row evolves as $\mu_{n+1}=\mu_nP_n$. The rules are defined for every starting state; ordinary conditional probabilities identify them whenever the corresponding conditioning state has positive probability.
<!-- bilingual-en:end -->

指定一条链，需要给出**初始分布** $\mu_0(i)=\Pr(X_0=i)$，以及第 $n$ 步从 $i$ 到 $j$ 的转移规则 $p_n(i,j)$。每一行都必须非负、加起来等于一。采用[[Markov矩阵左右约定|行随机约定]]，这些概率组成 $P_n$，概率行按 $\mu_{n+1}=\mu_nP_n$ 更新。模型为每个出发状态指定规则；当相应条件状态的概率为正时，它与普通条件概率相符。

<!-- bilingual-en:start -->
The [[概率乘法法则|multiplication rule]] then gives the probability of a full finite path:
<!-- bilingual-en:end -->

由[[概率乘法法则]]，一段完整有限路径的概率是

$$\Pr(X_0=i_0,\ldots,X_m=i_m)
=\mu_0(i_0)\prod_{n=0}^{m-1}p_n(i_n,i_{n+1}).$$

<!-- bilingual-en:start -->
The product follows the path one transition at a time; it does not assume independent states. A zero initial or transition probability makes the path probability zero without conditioning on an impossible path. These joint probabilities determine the [[有限维分布|finite-dimensional distributions]], including observations at nonconsecutive times after summing out the intervening states.
<!-- bilingual-en:end -->

这个乘积沿路径逐步计算，并不要求各时刻状态独立。初始概率或某个转移概率为零时，路径概率就是零，无需在一条不可能的路径上做条件除法。这些联合概率确定[[有限维分布]]；若只观察几个不连续时点，把中间状态求和消去即可。

> [!example] Sunny, then rainy, then rainy · 晴、雨、雨这条路径
> <!-- bilingual-en:start -->
> Suppose the starting probabilities for sunny/rainy weather are $(0.6,0.4)$. Use the row matrix below at every step: a sunny day is followed by rain with probability $0.1$, and a rainy day remains rainy with probability $0.6$.
> <!-- bilingual-en:end -->
>
> 设晴、雨的初始概率为 $(0.6,0.4)$，每步都用下面的行随机矩阵：晴天之后下雨的概率是 $0.1$，雨天之后仍下雨的概率是 $0.6$。
>
> $$P=\begin{pmatrix}0.9&0.1\\0.4&0.6\end{pmatrix},\qquad
> \Pr(X_0=S,X_1=R,X_2=R)=0.6\times0.1\times0.6=0.036.$$
>
> <!-- bilingual-en:start -->
> Here $S,R$ label sunny and rainy. The model says that tomorrow's forecast, conditional on today, is unchanged by learning yesterday. It does not say that rain tomorrow has the same unconditional probability as rain today.
> <!-- bilingual-en:end -->
>
> 这里 $S,R$ 分别标记晴、雨。模型说的是已知今天后，昨天的记录不再改变明天的预测；它没有说今天与明天下雨的无条件概率相同。

<!-- bilingual-en:start -->
## Two conventions to keep explicit
<!-- bilingual-en:end -->
*读教材时需要明确的两个约定*

<!-- bilingual-en:start -->
**Time homogeneity is an additional condition here.** A general chain may use a different $P_n$ at each step; a [[DTMC时间齐次转移核|time-homogeneous chain]] uses one fixed matrix. Thus a weather model may still be Markov when its known seasonal transition rule changes. Some textbooks, including the opening definition in Gallager's chapter, restrict “Markov chain” to the homogeneous case before discussing the more general one.
<!-- bilingual-en:end -->

**本库把时间齐次作为附加条件。** 一般链可以在每步使用不同的 $P_n$；[[DTMC时间齐次转移核|时间齐次链]]才使用同一个矩阵。所以，天气模型的转移规律可以随已知季节改变，同时仍具有 Markov 性。有的教材，包括 Gallager 该章开头的定义，会先把“Markov 链”限定为齐次情形，再讨论一般情况。

<!-- bilingual-en:start -->
**The finite/countable-state scope is also a convention.** Broader probability texts use “Markov chain” for discrete-time processes on continuous state spaces as well. An hourly real-valued measurement is outside this card's countable-state scope, but it can be a Markov chain in that broader usage if it has the Markov property. Discrete observation times alone establish neither version of the property.
<!-- bilingual-en:end -->

**有限或可数状态也属于范围约定。** 更一般的概率教材也把连续状态空间上的离散时间过程称为“Markov 链”。每小时取任意实数的观测不属于本卡的可数状态范围，但若满足 Markov 性，在广义用法中仍可称为 Markov 链。仅有离散观测时点，不能证明任何一种口径下的 Markov 性。

## 来源与核验

<!-- bilingual-en:start -->
- [Gallager, MIT 6.262 Chapter 3, PDF pp. 1–3](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/3558b08622765d26c2b0a7d2eeeac885_MIT6_262S11_chap03.pdf): supports finite/countable states, the one-step conditional condition, initial law, nonhomogeneous generalisation and state selection.
- [Weber, Cambridge Markov Chains, §1.2, PDF p. 6 / printed p. 2](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=6): Theorem 1.3 gives the path-product characterisation for a fixed matrix. The displayed time-dependent product applies the same conditional multiplication step with $P_n$; the weather calculation is a direct instance.
- [Pitman, Berkeley Stat205B Lecture 1, PDF pp. 1–2](https://www.stat.berkeley.edu/~pitman/s205s03/lecture1.pdf): confirms the broader state-space convention and kernels indexed by time. It is used to qualify terminology, not to impose a different scope on the local course.
<!-- bilingual-en:end -->

- Gallager 第 1–3 页核对可数状态、一步条件、初始律、非齐次推广和状态选择。Weber 第 1.2 节（PDF 第 6 页）定理 1.3 核对固定矩阵的路径乘积；本卡按同一条件乘法写出随时间变化的版本，并直接算天气例。Pitman 第 1–2 页核对一般状态空间及带时间下标的核，用于说明教材术语的范围差异。

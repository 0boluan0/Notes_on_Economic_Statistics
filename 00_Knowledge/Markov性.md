---
aliases:
- Markov property
- Markov 性
- 马尔可夫性
- 无后效性
student_os: knowledge-atom
atom_id: PROB-DTMC-001
atom_set: discrete-time-markov-chains
atom_type: definition
status: source-checked
requires:
- '[[条件独立]]'
- '[[随机过程]]'
related:
- '[[Markov充分状态]]'
- '[[DTMC时间齐次转移核]]'
- '[[CTMC时间齐次转移函数]]'
leads_to:
- '[[离散时间Markov链]]'
- '[[CTMC定义]]'
part_of:
- '[[离散时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Markov property
<!-- bilingual-en:end -->
*Markov 性（马尔可夫性）*

<!-- bilingual-en:start -->
A process has the **Markov property** when its current state contains all the information from its own history that is needed to predict its future. Imagine two weather records with different pasts but the same weather today. In a Markov weather model, learning which record occurred does not change tomorrow's forecast once today's state is known. This is a claim about the model's conditional probabilities, not a claim that actual weather necessarily behaves this way.
<!-- bilingual-en:end -->

过程具有 **Markov 性**，是说当前状态已经保留了自身历史中预测未来所需要的信息。想象两份过去不同、但今天天气相同的记录：在 Markov 天气模型里，知道今天的状态后，再知道实际是哪份历史，不会改变对明天的预测。这是在描述模型的条件概率，不能据此断言真实天气一定满足它。

<!-- bilingual-en:start -->
A [[随机过程|stochastic process]] records uncertain states at different times. Write $X_n$ for the state at step $n$. A [[条件概率|conditional probability]] describes the forecast after specified information is known; the information appears after the vertical bar. The Markov question compares two forecasts: one using the entire observed path, the other using only its last state, at the same time $n$.
<!-- bilingual-en:end -->

[[随机过程]]记录不同时刻的不确定状态。用 $X_n$ 表示第 $n$ 步的状态；[[条件概率]]表示知道指定信息之后的预测，已知信息写在竖线后。Markov 性比较的是同一时点 $n$ 的两种预测：一种使用完整历史，另一种只使用历史末端的当前状态。

> [!abstract] The one-step condition · 一步条件
> <!-- bilingual-en:start -->
> For a discrete-time process with finite or countable state space, every history with positive probability must satisfy, for every possible next state $j$,
> <!-- bilingual-en:end -->
>
> 对离散时间、有限或可数状态的过程，每段具有正概率的历史，都必须对任意下一状态 $j$ 满足
>
> $$\Pr(X_{n+1}=j\mid X_0=i_0,\ldots,X_n=i)
> =\Pr(X_{n+1}=j\mid X_n=i).$$
>
> <!-- bilingual-en:start -->
> The left side knows all earlier states; the right side knows only the current state. Requiring equality for **every** such history matters: checking one realised path, or only a few earlier observations, does not establish the property. Ordinary event-conditioning cannot divide by the probability of a zero-probability history.
> <!-- bilingual-en:end -->
>
> 左边知道全部早期状态，右边只知道当前状态。条件要求**所有**这样的历史都成立：检查一条实现的路径，或只检查少数早期观测，都不足以证明。对零概率历史，不能套用事件条件概率的除法公式。

<!-- bilingual-en:start -->
## Why this also describes the whole future
<!-- bilingual-en:end -->
*为什么也可以说“整个未来只看现在”*

<!-- bilingual-en:start -->
For two future steps, the [[概率乘法法则|multiplication rule]] gives the probability of a specified continuation by multiplying the next-step chance by the following transition chance. Applying the Markov condition at both steps removes the old path from these factors. Repeating this argument handles any finite future block. In discrete time, this determines the conditional law of the future sequence. Thus the future and the earlier path are [[条件独立|conditionally independent]] given the current state: once that state is known, the earlier path adds no information about any future event.
<!-- bilingual-en:end -->

对于未来两步，[[概率乘法法则]]把一条指定后续路径的概率写成“下一步概率”乘“再下一步转移概率”。在两步分别应用 Markov 条件，旧历史就从这些因子中消失。反复这样做，可以处理任意有限的未来片段；在离散时间里，这些概率确定整个未来序列的条件律。因此，也可以说未来与更早的历史在给定当前状态后[[条件独立]]：知道当前状态以后，更早的历史不再为任何未来事件补充信息。

> [!note]- Information notation for general time · 一般时间下的信息记号
> <!-- bilingual-en:start -->
> A [[σ-代数|sigma-algebra]] collects the events distinguishable from some information. Let $\mathcal F_t^-=\sigma(X_s:s<t)$ collect information from the earlier path, and $\mathcal F_t^+=\sigma(X_u:u>t)$ collect information from the later path. Relative to the process's own history, the Markov property can be written as
> <!-- bilingual-en:end -->
>
> [[σ-代数]]收集根据某份信息能够区分的事件。记 $\mathcal F_t^-=\sigma(X_s:s<t)$ 为较早路径包含的信息，$\mathcal F_t^+=\sigma(X_u:u>t)$ 为较晚路径包含的信息。相对于过程自身的历史，Markov 性可以写成
>
> $$\mathcal F_t^+\ \perp\!\!\!\perp\ \mathcal F_t^-\mid X_t.$$
>
> <!-- bilingual-en:start -->
> This means conditional independence of these two collections given the present. Conditional-distribution identities are interpreted almost surely, rather than by dividing by $\Pr(X_t=x)$ when that probability is zero. Adding external information to the history can change whether the property holds; the information being conditioned on must be specified.
> <!-- bilingual-en:end -->
>
> 它表示这两组信息在给定当前状态后条件独立。条件分布的等式按几乎处处成立来理解，不能在 $\Pr(X_t=x)=0$ 时直接相除。若给历史额外加入外部信息，Markov 性可能改变，所以必须说明以什么信息作为条件。

> [!example] Dependent states can still be Markov · 状态相关仍然可以是 Markov 的
> <!-- bilingual-en:start -->
> Toss one fair coin to choose $X_0\in\{0,1\}$, then set $X_{n+1}=X_n$ forever. Adjacent states are identical and are not independent: $\Pr(X_1=1\mid X_0=1)=1$, whereas $\Pr(X_1=1)=1/2$. Yet knowing the current state fixes the entire future, so earlier states cannot improve the forecast. The process is Markov.
> <!-- bilingual-en:end -->
>
> 抛一次公平硬币选定 $X_0\in\{0,1\}$，以后始终令 $X_{n+1}=X_n$。相邻状态完全相同，而且不独立：$\Pr(X_1=1\mid X_0=1)=1$，但 $\Pr(X_1=1)=1/2$。然而当前状态已经确定全部未来，更早的状态无法改善预测，因此过程具有 Markov 性。

<!-- bilingual-en:start -->
For a failure example, let $U,V$ be two independent fair bits and set $X_0=U$, $X_1=V$, and $X_n=U$ for all $n\ge2$. At step $1$, knowing $X_1$ alone gives $\Pr(X_2=1\mid X_1)=1/2$. Also revealing $X_0$ makes that probability either $0$ or $1$. The current state omitted predictive information, so this representation is not Markov. Choosing a [[Markov充分状态|sufficient state]] is therefore part of modelling, not merely naming the observations.
<!-- bilingual-en:end -->

再看一个不成立的例子：令 $U,V$ 为两个独立公平的二元结果，设 $X_0=U$、$X_1=V$，并对所有 $n\ge2$ 令 $X_n=U$。在第 $1$ 步，只知道 $X_1$，有 $\Pr(X_2=1\mid X_1)=1/2$；再告诉你 $X_0$，这个概率就变成 $0$ 或 $1$。当前状态遗漏了预测信息，所以这种表示不满足 Markov 性。选择[[Markov充分状态|充分状态]]本身就是建模的一部分。

<!-- bilingual-en:start -->
Here, “memorylessness” does not mean that history never happened or that successive states are independent. It means that the chosen present state retains the relevant history. It also does not require the rule to stay the same tomorrow: that separate condition is [[DTMC时间齐次转移核|time homogeneity]].
<!-- bilingual-en:end -->

“无后效性”并不是说历史没有发生过，也不是说相邻状态彼此独立，而是说选定的当前状态已经保留了相关历史。它也不要求明天仍采用同一规则；那是另外的[[DTMC时间齐次转移核|时间齐次性]]条件。

## 来源与核验

<!-- bilingual-en:start -->
- [Gallager, MIT 6.262 Chapter 3, PDF pp. 1–3](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/3558b08622765d26c2b0a7d2eeeac885_MIT6_262S11_chap03.pdf): Definition 3.1.1 and (3.3) support the one-step property, time-dependent kernels and the role of state selection. This chapter initially assumes homogeneity; that extra assumption is separated here.
- [Pitman, Berkeley Stat205B Lecture 1, p. 1](https://www.stat.berkeley.edu/~pitman/s205s03/lecture1.pdf): supports the past/present/future definition and the distinction between a Markov kernel and a homogeneous one.
- [Kemp, UCSD Math 280B, The Markov Property, PDF pp. 3 and 7](https://mathweb.ucsd.edu/~tkemp/280B/36.1.Markov-Property-After.pdf#page=3): supports the full-future sigma-field and conditional-independence formulation relative to the specified history. The coin examples are direct calculations of the stated definition.
- [[01_Math/05_随机过程/02_随机过程的概念和分类.docx|Local course terminology]]: checked only for the local synonym translated here as “memorylessness”.
<!-- bilingual-en:end -->

- Gallager 第 1–3 页核对一步条件、随时间变化的核与状态选择；其开头默认齐次，本卡把该附加条件分开。Pitman 第 1 页核对过去与未来的条件独立表述；Kemp 第 3、7 页核对完整未来信息与所选历史的关系。两个硬币例子直接按定义计算。本地课程文件仅核对“无后效性”术语。

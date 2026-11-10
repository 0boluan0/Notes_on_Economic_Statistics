---
aliases:
  - Non-Markovian observations of a hidden Markov model
  - 隐状态与观测的Markov性区别
student_os: knowledge-atom
atom_id: PROB-DTMC-043
atom_set: discrete-time-markov-chains
atom_type: implication-boundary
status: source-checked
requires:
  - "[[Markov switching模型]]"
  - "[[Markov性]]"
related:
  - "[[Markov充分状态]]"
  - "[[状态扩充]]"
leads_to: []
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# When hidden Markov observations retain memory
<!-- bilingual-en:end -->

*潜在状态的 Markov 性与观测的记忆*

<!-- bilingual-en:start -->
A noisy reading is only partial information about the system that produced it. Earlier readings can therefore help interpret the latest reading and improve the next prediction. In a [[Markov switching模型|Markov-switching model]], the hidden state can be first-order Markov while the observed sequence is not.

The [[Markov性|first-order Markov property]] for observations would mean that their next conditional distribution depends only on the latest observation, not on the earlier observed history. To disprove it, we need two histories with the same latest observation but different next-observation probabilities. The following example computes those probabilities explicitly.
<!-- bilingual-en:end -->

带噪读数只提供了系统状态的一部分信息。因此，更早的读数仍可能帮助解释最新读数，并改善下一步预测。在 [[Markov switching模型]] 中，潜在状态可以是一阶 Markov 链，而观测序列不是。

观测的 [[Markov性|一阶 Markov 性]] 要求：下一次观测的条件分布只依赖最新观测，不再依赖更早的观测历史。要反驳它，只需找到两段最新观测相同、下一次观测概率却不同的历史。下面把这两个概率完整算出。

<!-- bilingual-en:start -->
## A persistent state with an imperfect sensor

Let $S_t$ be a hidden state taking values $0$ or $1$. Its initial probabilities are $1/2,1/2$, and its [[Markov矩阵左右约定|row transition matrix]] is
<!-- bilingual-en:end -->

## 持续的状态与不完美的传感器

令潜在状态 $S_t$ 取值为 $0$ 或 $1$，初始概率各为 $1/2$。其 [[Markov矩阵左右约定|行转移矩阵]] 为

$$P=\begin{pmatrix}0.9&0.1\\0.1&0.9\end{pmatrix}.$$

<!-- bilingual-en:start -->
Thus each state remains unchanged with probability $0.9$. This initial distribution is a [[Markov稳态分布|stationary distribution]]: multiplying $(1/2,1/2)$ by $P$ gives $(1/2,1/2)$ again. The binary reading $Y_t$ matches $S_t$ with probability $0.8$ and reports the other value with probability $0.2$. Given the entire state path, assume the readings are [[条件独立|conditionally independent]], each depending only on its current state. This completes the observation model; it is stronger than specifying the one-time error rate alone.
<!-- bilingual-en:end -->

因此，每个状态都以 $0.9$ 的概率保持不变。这个初始分布是 [[Markov稳态分布|平稳分布]]：$(1/2,1/2)$ 右乘 $P$ 后仍为 $(1/2,1/2)$。二元读数 $Y_t$ 以 $0.8$ 的概率等于 $S_t$，以 $0.2$ 的概率报出另一个值。再假设：给定整个状态路径后，各次读数 [[条件独立]]，每次只依赖当期状态。这才完整指定了观测模型，比仅给出每次误报概率更强。

<!-- bilingual-en:start -->
## Use the first reading, then update with the second

Let $q_1=\Pr(S_1=1\mid Y_1)$ be our probability for state $1$ after the first reading. Applying [[Bayes法则|Bayes' rule]] to the equally likely initial states gives
<!-- bilingual-en:end -->

## 先利用第一条读数，再用第二条更新

令 $q_1=\Pr(S_1=1\mid Y_1)$ 表示看到第一条读数后，对状态 $1$ 的概率判断。对初始等概率的两个状态使用 [[Bayes法则]]，得到

$$
q_1=
\begin{cases}
\dfrac{0.8(1/2)}{0.8(1/2)+0.2(1/2)}=0.8,&Y_1=1,\\[6pt]
\dfrac{0.2(1/2)}{0.2(1/2)+0.8(1/2)}=0.2,&Y_1=0.
\end{cases}
$$

<!-- bilingual-en:start -->
Before the second reading arrives, the [[全概率公式|law of total probability]] combines the two possible starting states. Write $r_2=\Pr(S_2=1\mid Y_1)$ for this one-step prediction:
<!-- bilingual-en:end -->

在第二条读数到来前，[[全概率公式]] 把两种起始状态的可能性合在一起。用 $r_2=\Pr(S_2=1\mid Y_1)$ 表示这个一步预测：

$$r_2=0.9q_1+0.1(1-q_1)=0.1+0.8q_1.$$

<!-- bilingual-en:start -->
So $r_2=0.74$ after $Y_1=1$, but $r_2=0.26$ after $Y_1=0$. Now suppose the second reading is $Y_2=1$ in both cases. Write $q_2=\Pr(S_2=1\mid Y_1,Y_2=1)$ for the updated state probability. Another Bayes update gives
<!-- bilingual-en:end -->

因此，$Y_1=1$ 时 $r_2=0.74$；$Y_1=0$ 时 $r_2=0.26$。现在令两种情况下的第二条读数都为 $Y_2=1$。用 $q_2=\Pr(S_2=1\mid Y_1,Y_2=1)$ 表示更新后的状态概率，再用一次 Bayes 公式：

$$
q_2=\frac{0.8r_2}{0.8r_2+0.2(1-r_2)}
=\begin{cases}
148/161,&(Y_1,Y_2)=(1,1),\\
52/89,&(Y_1,Y_2)=(0,1).
\end{cases}
$$

<!-- bilingual-en:start -->
Two consecutive readings of $1$ make state $1$ more plausible than a reading of $0$ followed by $1$. The latest reading has not erased what we learned from the earlier one.
<!-- bilingual-en:end -->

连续两次读到 $1$，比先读到 $0$ 再读到 $1$ 更支持当前处于状态 $1$。最新读数并没有抹掉早先读数提供的信息。

<!-- bilingual-en:start -->
## The next reading has two different probabilities

Given either history, the probability of $S_3=1$ is $0.1+0.8q_2$. A reading of $1$ then has probability $0.8$ in state $1$ and $0.2$ in state $0$. Combining these steps gives
<!-- bilingual-en:end -->

## 下一条读数出现两个不同的概率

给定其中任一历史，$S_3=1$ 的概率是 $0.1+0.8q_2$。随后，在状态 $1$ 中读到 $1$ 的概率为 $0.8$，在状态 $0$ 中为 $0.2$。合并这两步：

$$
\begin{aligned}
\Pr(Y_3=1\mid Y_1,Y_2=1)
&=0.8(0.1+0.8q_2)+0.2\bigl[1-(0.1+0.8q_2)\bigr]\\
&=0.26+0.48q_2.
\end{aligned}
$$

> [!example] Same latest reading, different predictions / 最新读数相同，预测不同
> $$
> \Pr(Y_3=1\mid Y_1=1,Y_2=1)=\frac{1129}{1610}\approx0.70124,
> $$
> $$
> \Pr(Y_3=1\mid Y_1=0,Y_2=1)=\frac{481}{890}\approx0.54045.
> $$
> <!-- bilingual-en:start -->
> The conditioning histories have positive probabilities $161/500$ and $89/500$, respectively. Both end with $Y_2=1$, yet their predictions differ. Hence this observed sequence is not first-order Markov.
> <!-- bilingual-en:end -->
>
> 两个条件历史的概率分别为 $161/500$ 和 $89/500$，都大于零。它们都以 $Y_2=1$ 结束，预测却不同。因此，这个观测序列不满足一阶 Markov 性。

<!-- bilingual-en:start -->
## Where the predictive information lives

The relevant summary is not necessarily the last reading. In this two-state model, $q_t=\Pr(S_t=1\mid Y_1,\ldots,Y_t)$ summarizes the state uncertainty after all readings so far. With the model parameters known, predicting one step and applying Bayes' rule updates this probability using only $q_t$ and the next reading. That is the idea behind filtering: carry forward a probability distribution over the hidden state, rather than pretend that its noisy measurement is the state itself. For the general question of what information a state must retain, see [[Markov充分状态|sufficient states]] and [[状态扩充|state augmentation]].

The conclusion is “need not”, not “never”. If the sensor reveals the state exactly, $Y_t=S_t$ is Markov. If both states have the same emission distribution in the conditionally independent model above, the observations are [[独立同分布|independent and identically distributed]], even though the hidden states may be dependent. What fails is the automatic inference from a Markov hidden state to Markov observations.
<!-- bilingual-en:end -->

## 预测信息究竟保存在哪里？

合适的摘要不一定是最后一条读数。在这个两状态模型中，$q_t=\Pr(S_t=1\mid Y_1,\ldots,Y_t)$ 概括了看到截至目前所有读数后，对状态的不确定性。模型参数已知时，通过一步预测和 Bayes 更新，只使用 $q_t$ 与下一条读数就能更新这个概率。这就是滤波的想法：持续保留潜在状态的概率分布，而不是把带噪测量当作状态本身。状态究竟需要保存哪些信息，见 [[Markov充分状态]] 与 [[状态扩充]]。

结论是“不一定”，不是“永远不可能”。如果传感器准确显示状态，$Y_t=S_t$ 就是 Markov 链。如果两个状态的发射分布完全相同，而且仍采用上述给定状态路径后条件独立的模型，那么观测是 [[独立同分布]] 的，尽管潜在状态之间可能相关。不能成立的是“潜在状态 Markov，所以观测自动 Markov”的推断。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Rabiner (1989), A Tutorial on Hidden Markov Models](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf), printed pp. 259–262 (PDF pp. 3–6): the hidden-state and emission structure, conditional path factorization, and summation over unobserved states. The binary counterexample and its fractions are direct calculations from this card's stated model.
- [Hamilton (2005), Regime-Switching Models](https://econweb.ucsd.edu/~jhamilton/palgrav1.pdf), printed pp. 3–5 (PDF pp. 4–6), especially equations (6)–(9): state probabilities conditional on observed history and their prediction/update steps.
<!-- bilingual-en:end -->

- Rabiner，印刷第 259–262 页（PDF 第 3–6 页）：核对潜在状态与发射结构、条件路径分解，以及对未观测状态求和。本文二元反例及各分数由明示模型直接计算。
- Hamilton，印刷第 3–5 页（PDF 第 4–6 页），特别是式 (6)–(9)：核对以观测历史为条件的状态概率，以及预测与更新步骤。

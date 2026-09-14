---
aliases:
  - "齐次 Poisson 计数是出生率恒定的纯生链"
  - A homogeneous Poisson count is a constant-rate pure-birth chain
  - Poisson process as a pure-birth CTMC
  - 泊松过程的纯生链表示
student_os: knowledge-atom
atom_id: PROB-CTMC-021
atom_set: continuous-time-markov-chains
atom_type: equivalence
status: source-checked
mastery_state: unassessed
requires:
  - "[[纯生链]]"
  - "[[齐次泊松过程]]"
related:
  - "[[泊松指数等价]]"
  - "[[非齐次泊松过程]]"
  - "[[生灭链]]"
leads_to:
  - "[[CTMC共同率估计]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 齐次 Poisson 计数是出生率恒定的纯生链
<!-- bilingual-en:start -->
*A homogeneous Poisson counting process is a pure-birth chain with a constant birth rate*
<!-- bilingual-en:end -->

> [!summary] 两种描述同一个过程
> 从 $N(0)=0$ 开始，若[[纯生链]]在每一层都有同一个出生率 $\lambda>0$，即
> $$
> q_{n,n+1}=\lambda,\qquad q_{nn}=-\lambda,\qquad n\ge0,
> $$
> 则 $N$ 是率为 $\lambda$ 的[[齐次泊松过程]]；反过来，齐次 Poisson 计数的生成率正是上式。
> <!-- bilingual-en:start -->
> A homogeneous Poisson count and a constant-rate pure-birth CTMC are two descriptions of the same process.
> <!-- bilingual-en:end -->

常数率使每段停留时间独立同分布为 $\operatorname{Exp}(\lambda)$，所以
$$
N(t)\sim\operatorname{Poisson}(\lambda t),
$$
并具有平稳、独立增量。这里“常速”指 $\lambda_n$ 不随当前计数 $n$ 改变。

若改成状态依赖率 $q_{n,n+1}=\lambda_n$，模型仍是时间齐次纯生链，但通常不再有平稳、独立增量，而且还要另查是否爆炸。若率依赖日历时点 $\lambda(t)$，得到的是[[非齐次泊松过程]]：增量仍独立但不平稳，生成率本身随时间改变。状态依赖与日历时间依赖是两种不同变化。

> [!example] 随计数变快
> 若 $\lambda_n=(n+1)\lambda$，下一次等待随当前计数增加而缩短。这是非爆炸的状态依赖纯生链，因为 $\sum_n1/[(n+1)\lambda]=\infty$；它不是率为 $\lambda$ 的齐次 Poisson 计数。

> [!question]- 自检
> 一个计数过程从 $n$ 到 $n+1$ 的速率为 $n+2$。它是否是时间齐次 CTMC？是否是齐次 Poisson 过程？
>
> **答案：** 它是时间齐次纯生 CTMC，并由倒数率级数发散可知非爆炸；但它不是齐次 Poisson 过程，因为出生率依赖当前计数。

## 来源与核验

- [Cambridge Applied Probability notes, §§1.4–1.5](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对 homogeneous Poisson process 的常速纯生构造与一般 birth process 的状态依赖率。
- [James Norris, Markov Chains, §§2.4–2.5](https://www.statslab.cam.ac.uk/~jrn10/Markov/)：核对 Poisson processes 与 birth processes 的等价接口。
- [[齐次泊松过程]]与[[泊松指数等价]]：分别给出计数定义和指数间隔刻画；常数出生率把这两种表述连到纯生链。

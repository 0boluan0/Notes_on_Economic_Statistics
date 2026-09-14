---
aliases:
  - "DTMC 的时间齐次转移核是不随起始时点改变的状态转移规律"
  - "DTMC 时间齐次只表示同一转移核不随时点改变"
  - Time-homogeneous DTMC
  - Homogeneous DTMC transition kernel
  - 离散时间齐次马尔可夫链
student_os: knowledge-atom
atom_id: PROB-DTMC-003
atom_set: discrete-time-markov-chains
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[离散时间Markov链]]"
related:
  - "[[Markov性]]"
  - "[[Markov矩阵]]"
  - "[[齐次不等于平稳]]"
  - "[[CTMC时间齐次转移函数]]"
leads_to:
  - "[[Markov矩阵左右约定]]"
  - "[[DTMC转移复合]]"
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# DTMC 的时间齐次转移核是不随起始时点改变的状态转移规律
<!-- bilingual-en:start -->
*A time-homogeneous DTMC uses a transition law that does not change with the starting time*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 对离散时间 Markov 链，时间齐次是指存在同一组一步转移概率 $p_{ij}$，使每个时点 $n$ 都满足
> $$
> \Pr(X_{n+1}=j\mid X_n=i)=p_{ij}.
> $$
> 更一般地，经过 $m$ 步的转移概率只依赖间隔 $m$：
> $$
> \Pr(X_{n+m}=j\mid X_n=i)=p_{ij}^{(m)},
> $$
> 而不依赖从哪个 $n$ 开始。
> <!-- bilingual-en:start -->
> A DTMC is time homogeneous when the transition law over a fixed number of steps is invariant to the calendar time at which the interval begins.
> <!-- bilingual-en:end -->

Markov 性与时间齐次回答不同问题。Markov 性问过去在给定现在后是否还影响未来；时间齐次则在已经给定当前状态后，继续问同一种转移规则是否适用于每个时点。非齐次 Markov 链仍然可以只依赖当前状态，但要用随时间变化的转移核 $P_n$。
<!-- bilingual-en:start -->
The Markov property and time homogeneity answer different questions. The Markov property asks whether the past still affects the future once the present is known. Time homogeneity asks whether the same transition rule applies at every calendar time. A nonhomogeneous Markov chain can still depend only on its current state, but it uses a time-varying transition kernel $P_n$.
<!-- bilingual-en:end -->

> [!example] 同一设备，不同季节
> 若设备从“正常”到“故障”的下一小时概率全年保持相同，可以使用固定转移核。若高温月份的故障概率更高，即使当前状态仍足以预测下一步，转移核也随日历时间变化，链不是时间齐次的。
> <!-- bilingual-en:start -->
> If the probability of moving from “working” to “failed” during the next hour is constant throughout the year, one fixed transition kernel is appropriate. If failure is more likely in hot months, the current state may still be sufficient for predicting the next step, but the kernel changes with calendar time and the chain is not time homogeneous.
> <!-- bilingual-en:end -->

> [!question]- 自检
> 下一步只依赖当前状态，但周末与工作日采用不同转移概率。这条链有 Markov 性吗？时间齐次吗？
> <!-- bilingual-en:start -->
> The next state depends only on the current state, but weekends and weekdays use different transition probabilities. Is the chain Markov? Is it time homogeneous?
> <!-- bilingual-en:end -->
>
> **答案：** 可以有 Markov 性，但不是时间齐次的。
> <!-- bilingual-en:start -->
> **Answer:** It can satisfy the Markov property, but it is not time homogeneous.
> <!-- bilingual-en:end -->

## 来源与核验

- [MIT OCW 6.262, Chapter 3, equations (3.2)–(3.3)](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/3558b08622765d26c2b0a7d2eeeac885_MIT6_262S11_chap03.pdf#page=2)：核对齐次与非齐次转移概率的区分。
- [Cambridge Markov Chains notes, §1.3](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=3)：核对固定转移矩阵与按时间间隔定义的多步转移概率。

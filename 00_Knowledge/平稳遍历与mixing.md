---
aliases:
  - "平稳性遍历性与 mixing 描述不同层次且不能互作同义词"
  - Stationarity ergodicity and mixing hierarchy
  - Mixing implies ergodicity
student_os: knowledge-atom
atom_id: TS-STAT-021
atom_set: stationarity-ergodicity-spectrum
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[严平稳定义]]"
related:
  - "[[均值遍历]]"
  - "[[协方差遍历]]"
  - "[[随机常数非遍历反例]]"
part_of:
  - "[[平稳性、遍历性与谱.canvas]]"
---

# 平稳性遍历性与 mixing 描述不同层次且不能互作同义词
<!-- bilingual-en:start -->
*Stationarity, ergodicity, and mixing describe different levels and are not synonyms*
<!-- bilingual-en:end -->

> [!summary] 原子区分
> 平稳性说概率律对时间平移不变；完整遍历性通常说平移不变事件只能有概率 0 或 1，从而一条典型路径的广泛时间平均可由遍历定理恢复总体；mixing 进一步要求相隔越来越远的事件渐近解耦。标准 stationary shift 语境下，strong mixing 推出 weak mixing，再推出 ergodicity，反向一般不成立。
> <!-- bilingual-en:start -->
> Stationarity is invariance of the law under shifts. Full ergodicity concerns triviality of shift-invariant events and supports broad time-average results. Mixing additionally requires increasingly separated events to become asymptotically decoupled. In the standard stationary-shift setting, strong mixing implies weak mixing, which implies ergodicity; the converses fail in general.
> <!-- bilingual-en:end -->

术语约定并不完全统一：有的教材只对平稳、保测的 shift 定义 ergodic，有的先独立定义二者。因此写结论时必须给出采用的定义和前提，不能把“相关衰减”“均值遍历”“完整遍历”或“mixing”互换。均值遍历只针对一个统计量，远弱于完整遍历。
<!-- bilingual-en:start -->
Conventions vary: some texts define ergodicity only for a stationary measure-preserving shift, while others state stationarity separately. Always name the definition and assumptions. Correlation decay, mean ergodicity, full ergodicity, and mixing are not interchangeable; mean ergodicity concerns only one statistic.
<!-- bilingual-en:end -->

> [!question]- 自检
> 某过程的样本均值能恢复总体均值，能否据此就把它称为 strong mixing？
>
> **答案：** 不能。均值遍历只针对一个统计量；strong mixing 要求越来越远的事件渐近解耦，是强得多的整体依赖条件。

## 来源与核验

- [MIT 6.441 Information Theory course notes, Definitions 8.2–8.3](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/5d8f16adc3385c9ff2975b121bd620e4_MIT6_441S16_course_notes.pdf)：核对 stationarity、shift-invariant-event ergodicity、strong/weak mixing 定义与蕴含链，并核对术语边界。
- [UC Berkeley Statistics 205B, Lecture 9](https://www.stat.berkeley.edu/~pitman/s205s03/lecture9.pdf)：核对 stationary process 与 invariant sigma-field 的测度论框架。
<!-- bilingual-en:start -->
- MIT 6.441 was checked for separate stationarity and ergodicity definitions and the strong-mixing to weak-mixing to ergodicity implication chain.
- Berkeley Statistics 205B was checked for the stationary-shift and invariant-sigma-field framework.
<!-- bilingual-en:end -->

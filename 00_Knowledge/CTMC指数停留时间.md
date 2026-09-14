---
aliases:
  - "齐次纯跳 CTMC 的非吸收状态停留时间服从指数分布"
  - Exponential holding time
  - CTMC sojourn time
  - CTMC dwell time
  - 指数停留时间
student_os: knowledge-atom
atom_id: PROB-CTMC-002
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[CTMC定义]]"
  - "[[CTMC时间齐次转移函数]]"
related:
  - "[[Markov充分状态]]"
  - "[[状态扩充]]"
  - "[[普通更新模型边界]]"
leads_to:
  - "[[出口率]]"
  - "[[生成矩阵分解]]"
  - "[[有界率非爆炸]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 齐次纯跳 CTMC 的非吸收状态停留时间服从指数分布
<!-- bilingual-en:start -->
*A nonabsorbing state of a homogeneous pure-jump CTMC has an exponential holding time*
<!-- bilingual-en:end -->

> [!summary] 核心结论
> 对标准时间齐次纯跳 CTMC，进入状态 $i$ 后的停留时间 $H_i$ 满足无记忆性。若总离开率 $0<q_i<\infty$，则
> $$
> \Pr(H_i>t)=e^{-q_it},\qquad H_i\sim\operatorname{Exp}(q_i),qquad \mathbb E H_i=\frac1{q_i}.
> $$
> 若 $q_i=0$，$i$ 是吸收态且 $H_i=\infty$ 几乎必然，不能写成普通的 $\operatorname{Exp}(0)$ 随机变量。
> <!-- bilingual-en:start -->
> Homogeneity and the Markov property make a finite-rate holding time memoryless. A zero exit rate means infinite holding, not an ordinary exponential random variable with parameter zero.
> <!-- bilingual-en:end -->

无记忆性来自
$$
\Pr(H_i>s+t\mid H_i>s)=\Pr(H_i>t).
$$
它要求当前状态已经包含决定未来离开危险率的全部信息。Markov 性本身若没有时间齐次，停留时间仍可依赖进入的日历时点；一般连续状态 Markov 过程也根本没有这种分段常值停留结构。

观察到非指数停留时间时，至少有三种可能：状态合并过粗，需加入阶段或年龄；速率随日历时间变化；过程本来就是 semi-Markov。不能只凭一张停留时间直方图断言“世界不 Markov”。

> [!example] 最小反例
> 若维修完成的危险率随已经维修的时长上升，维修时间可能服从 Weibull 分布。只保留“维修中”一个状态时，它不是齐次两状态 CTMC；加入维修年龄、拆成多个阶段或使用 semi-Markov 模型才与机制相符。

> [!question]- 自检
> 某状态的 $q_i=0$ 时，下一跳分布 $q_{ij}/q_i$ 是什么？
> <!-- bilingual-en:start -->
> If a state has $q_i=0$, what is its next-jump distribution $q_{ij}/q_i$?
> <!-- bilingual-en:end -->
>
> **答案：** 不定义。该状态不会离开，应把它作为吸收态单独处理。
> <!-- bilingual-en:start -->
> **Answer:** It is undefined. The process never leaves this state, so the state must be treated as absorbing.
> <!-- bilingual-en:end -->

## 来源与核验

- [MIT OCW 6.436J, Lecture 24, Proposition 1](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=6)：核对齐次 Markov 性推出停留时间无记忆与指数分布的论证。
- [Cambridge Applied Probability notes, jump chain and holding times](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对 $q_i=0$ 的吸收边界及 $q_i>0$ 时的停留时间参数。

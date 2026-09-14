---
aliases:
  - "CTMC 的时间齐次转移函数只依赖经过的时间而不依赖起始时点"
  - Time-homogeneous CTMC transition function
  - Homogeneous CTMC transition function
  - CTMC 齐次转移函数
  - 时间齐次 CTMC
student_os: knowledge-atom
atom_id: PROB-CTMC-037
atom_set: continuous-time-markov-chains
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[CTMC定义]]"
related:
  - "[[DTMC时间齐次转移核]]"
leads_to:
  - "[[CTMC指数停留时间]]"
  - "[[CTMC转移半群]]"
  - "[[生成矩阵约束]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# CTMC 的时间齐次转移函数只依赖经过的时间而不依赖起始时点
<!-- bilingual-en:start -->
*A time-homogeneous CTMC has a transition function that depends on elapsed time, not on the starting time*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 对连续时间 Markov 链，若从任意时点 $s$、任意状态 $i$ 出发，经过 $t\ge 0$ 后到达 $j$ 的条件概率都可写成
> $$
> \Pr(X_{s+t}=j\mid X_s=i)=p_{ij}(t),
> $$
> 且右侧与 $s$ 无关，就称这条链**时间齐次**。把所有 $p_{ij}(t)$ 排成矩阵，得到随时间间隔变化的一族转移矩阵 $P(t)$。
> <!-- bilingual-en:start -->
> A CTMC is time homogeneous when, from any starting time $s$ and state $i$, the conditional probability of being in state $j$ after an elapsed time $t$ can be written as $p_{ij}(t)$, independently of $s$. These probabilities form a family of transition matrices $P(t)$.
> <!-- bilingual-en:end -->

时间齐次只去掉了对起始时点的依赖，并没有把连续时间压成“一步”。DTMC 通常由一个固定的一步转移矩阵反复迭代；CTMC 则要用整个矩阵族 $\{P(t):t\ge0\}$ 描述不同长度的时间间隔。$P(t)$ 怎样复合、何时能由生成矩阵微分得到，是建立在这个定义之上的进一步结论。
<!-- bilingual-en:start -->
Time homogeneity removes dependence on the starting time; it does not turn continuous time into a single step. A DTMC usually iterates one fixed one-step matrix, whereas a CTMC uses the whole family $\{P(t):t\ge0\}$. Composition and generator formulas are additional results built on this definition.
<!-- bilingual-en:end -->

> [!example] 单向故障
> 设备有“正常”与“故障”两个状态。正常设备以常数率 $\lambda$ 故障，故障后不再恢复，则
> $$
> P(t)=
> \begin{pmatrix}
> e^{-\lambda t}&1-e^{-\lambda t}\\
> 0&1
> \end{pmatrix}.
> $$
> 无论从上午还是下午开始观察，经过同样长度 $t$ 后的转移概率都相同，因此该模型时间齐次。若故障率随日历时点变化，就需要同时记录起始时点与终止时点的两参数转移函数。
> <!-- bilingual-en:start -->
> Suppose a working device fails at constant rate $\lambda$ and never recovers. Its transition matrix over an interval of length $t$ is shown above. The same elapsed time gives the same transition probabilities at every starting time, so the model is time homogeneous. A calendar-time-dependent failure rate instead requires a two-parameter transition function.
> <!-- bilingual-en:end -->

> [!question]- 自检 · Check
> 某设备从“正常”到“故障”的概率只取决于经过了多久，但从早晨 8 点开始的两小时与从晚上 8 点开始的两小时使用不同概率。它有时间齐次转移函数吗？
> <!-- bilingual-en:start -->
> A device uses different two-hour failure probabilities when the interval starts at 8 a.m. and at 8 p.m. Does it have a time-homogeneous transition function?
> <!-- bilingual-en:end -->
>
> **答案：** 没有。相同时间间隔的转移概率仍依赖起始时点。
> <!-- bilingual-en:start -->
> **Answer:** No. The transition probability over the same elapsed time still depends on the starting time.
> <!-- bilingual-en:end -->

## 来源与核验

- [MIT OCW 6.436J, Lecture 24, Definition 2](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=5)：核对时间齐次 CTMC 的条件概率只依赖 $t-s$，以及 $p_{ij}(t)$ 的记号。
- [Cambridge Applied Probability notes, §1.1](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=3)：核对时间齐次定义、转移矩阵族 $P(t)$ 与非负时间参数。
- [Cambridge Applied Probability notes, §1.3](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=6)：核对常数离开率对应的指数停留时间，用于上面的单向故障例子。

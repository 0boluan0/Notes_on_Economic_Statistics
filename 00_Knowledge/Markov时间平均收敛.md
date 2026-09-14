---
aliases:
  - "可数不可约正常返链的有界函数时间平均从任意初始分布出发都几乎必然收敛到平稳期望"
  - Markov chain ergodic time average
  - Markov chain ergodic theorem
  - 周期链时间平均
student_os: knowledge-atom
atom_id: PROB-DTMC-030
atom_set: discrete-time-markov-chains
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[正常返与零常返]]"
  - "[[不可约链]]"
  - "[[Markov稳态分布]]"
related:
  - "[[周期链Cesaro平均]]"
  - "[[Kac回返公式]]"
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# 可数不可约正常返链的有界函数时间平均从任意初始分布出发都几乎必然收敛到平稳期望
<!-- bilingual-en:start -->
*For a countable irreducible positive recurrent chain, time averages of bounded functions converge almost surely to the stationary expectation from any initial distribution*
<!-- bilingual-en:end -->

> [!summary] 样本路径的时间平均定理
> 设 DTMC 的状态空间 $S$ 可数，链不可约且正常返，唯一平稳分布为 $\pi$。对任意初始分布 $\mu$ 和任意有界函数 $f:S\to\mathbb R$，
> $$
> \frac1N\sum_{n=0}^{N-1}f(X_n)
> \xrightarrow{a.s.}
> \sum_{j\in S}\pi_jf(j).
> $$
> 这个结论不要求非周期。
> <!-- bilingual-en:start -->
> From any initial distribution, the sample-path average of a bounded function converges almost surely to its expectation under the stationary distribution. Aperiodicity is not required.
> <!-- bilingual-en:end -->

这里平均的是**同一条随机路径上实际观察到的值**。它不同于[[周期链Cesaro平均]]中的 $\frac1N\sum_n p_{ij}^{(n)}$：后者平均的是一列转移概率，前者本身是随机变量。

取二状态确定性交替链和 $f(x)=\mathbf 1\{x=1\}$。无论从哪个状态出发，一条路径在状态 1 的前 $N$ 步占比都趋于 $1/2$；与此同时，每一步的边际分布仍可随奇偶振荡。周期因此不妨碍路径时间平均收敛。

> [!question]- 自检
> 在有限不可约但周期为 2 的链上，能否用一条足够长的路径估计状态 $j$ 的平稳占比？
>
> **答案：** 可以。令 $f(x)=\mathbf 1\{x=j\}$，时间平均就是访问 $j$ 的经验比例，并几乎必然收敛到 $\pi_j$；不需要非周期。

## 来源与核验

- [Cambridge Markov Chains notes, Theorem 10.2](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=41)：核对任意初始分布下有界函数的遍历时间平均，以及该定理不要求非周期。
- [MIT OCW 6.262, Chapter 5, Theorem 5.1.2](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/01d0892549619cb25d928f15ec7230ed_MIT6_262S11_chap05.pdf#page=9)：以状态指示函数交叉核对长期访问比例。

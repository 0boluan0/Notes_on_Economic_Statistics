---
aliases:
  - "n 步转移由 Chapman-Kolmogorov 方程复合一步转移"
  - Chapman-Kolmogorov equation
  - n-step transition probabilities
  - C-K 方程
student_os: knowledge-atom
atom_id: PROB-DTMC-004
atom_set: discrete-time-markov-chains
atom_type: identity
status: source-checked
mastery_state: unassessed
requires:
  - "[[DTMC时间齐次转移核]]"
  - "[[Markov矩阵左右约定]]"
  - "[[全概率公式]]"
related:
  - "[[n步转移非首达]]"
  - "[[CTMC转移半群]]"
leads_to:
  - "[[马氏链沟通类]]"
  - "[[状态周期]]"
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# n 步转移由 Chapman-Kolmogorov 方程复合一步转移
<!-- bilingual-en:start -->
*The Chapman-Kolmogorov identity composes shorter transition intervals into an n-step transition*
<!-- bilingual-en:end -->

> [!summary] 核心恒等式
> 对时间齐次链，记
> $$
> p_{ij}^{(n)}=\Pr(X_{r+n}=j\mid X_r=i),
> $$
> 其中右侧不依赖起始时点 $r$。对任意 $m,n\ge0$，
> $$
> p_{ij}^{(m+n)}
> =\sum_k p_{ik}^{(m)}p_{kj}^{(n)}.
> $$
> 采用一致的转移矩阵约定后，这就是
> $$
> P^{(m+n)}=P^{(m)}P^{(n)},\qquad P^{(n)}=P^n,\qquad P^{(0)}=I.
> $$
> <!-- bilingual-en:start -->
> Conditioning on the intermediate state after $m$ steps gives the Chapman-Kolmogorov identity and, in a homogeneous chain, the matrix-power rule.
> <!-- bilingual-en:end -->

证明只需把第 $m$ 步可能出现的状态 $k$ 分成互斥且穷尽的情形。先从 $i$ 在 $m$ 步到达 $k$，再从 $k$ 用 $n$ 步到达 $j$；Markov 性使第二段在给定 $X_m=k$ 后不再依赖更早路径，最后对所有 $k$ 求和。

> [!example] 两步转移
> 两步后从 $i$ 到 $j$ 的概率为
> $$
> p_{ij}^{(2)}=\sum_k p_{ik}p_{kj}.
> $$
> 中间状态即使很多，也只需逐个计算“先到 $k$、再到 $j$”并相加。

> [!question]- 自检
> 为什么计算 $p_{ij}^{(m+n)}$ 时必须对中间状态 $k$ 求和？
>
> **答案：** 第 $m$ 步一定处于某个 $k$；这些中间状态事件互斥且穷尽，应用全概率公式即可。

## 来源与核验

- [Cambridge Markov Chains notes, §1.5](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=3)：核对 Chapman–Kolmogorov 求和式、$P^{(n)}=P^n$ 与 $P^{(0)}=I$。
- [MIT OCW 6.262, Chapter 3, §3.1](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/3558b08622765d26c2b0a7d2eeeac885_MIT6_262S11_chap03.pdf#page=3)：核对按中间状态分解路径概率。

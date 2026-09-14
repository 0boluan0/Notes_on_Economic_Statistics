---
aliases:
  - "n 步后位于目标状态不等于首次在第 n 步到达目标状态"
  - n-step occupancy is not first passage
  - n-step transition versus first passage
  - n 步转移非首达
student_os: knowledge-atom
atom_id: PROB-DTMC-027
atom_set: discrete-time-markov-chains
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[DTMC转移复合]]"
  - "[[Markov性]]"
related:
  - "[[吸收链]]"
leads_to:
  - "[[命中概率第一步方程]]"
  - "[[期望命中时间方程]]"
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# n 步后位于目标状态不等于首次在第 n 步到达目标状态
<!-- bilingual-en:start -->
*Being in a target state after n steps is not the same as first reaching it at step n*
<!-- bilingual-en:end -->

> [!summary] 两个概率事件
> 对 $i\ne j$，$n$ 步转移概率
> $$
> p_{ij}^{(n)}=\Pr_i(X_n=j)
> $$
> 只问第 $n$ 步是否位于 $j$。令
> $$
> \tau_j=\inf\{m\ge0:X_m=j\},\qquad
> f_{ij}^{(n)}=\Pr_i(\tau_j=n),
> $$
> 则首达概率还要求此前从未访问 $j$。因此两者一般不同。
> <!-- bilingual-en:start -->
> An n-step transition probability records occupancy at time n. A first-passage probability additionally excludes every path that reached the target earlier.
> <!-- bilingual-en:end -->

按第一次到达 $j$ 的时点分解，可得更新关系
$$
p_{ij}^{(n)}
=\sum_{m=1}^{n}f_{ij}^{(m)}p_{jj}^{(n-m)}
\qquad(i\ne j).
$$
每一项表示先在第 $m$ 步首次到达 $j$，再从 $j$ 运行剩余 $n-m$ 步后回到 $j$。这个分解也说明，直接读取 $(P^n)_{ij}$ 会把早到、离开后返回以及早到后停留的路径全部算进去。

> [!example] 吸收目标
> 若从 0 一步必到吸收状态 1，那么 $p_{01}^{(3)}=1$，因为第三步仍在 1；但 $f_{01}^{(3)}=0$，因为首次到达发生在第一步。

> [!question]- 自检
> 已知 $(P^5)_{ij}=0.4$，能否直接说“首次在第五步到达 $j$ 的概率是 0.4”？
>
> **答案：** 不能。还必须排除前四步已经访问过 $j$ 的路径。

## 来源与核验

- [MIT OCW 6.262, Chapter 5, §5.1](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/01d0892549619cb25d928f15ec7230ed_MIT6_262S11_chap05.pdf#page=4)：核对 $n$ 步占据概率、首次到达概率及其更新分解。
- [Cambridge Markov Chains notes, §3](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=10)：核对 hitting time 与 first-step 分析的记号。

---
aliases:
  - "Berge 定理用增广路刻画最大匹配"
  - Berge 定理
  - Berge's theorem
  - Augmenting-path criterion for maximum matching
student_os: knowledge-atom
atom_id: MCS-MATCH-007
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[增广路对称差]]"
part_of:
  - "[[二分图匹配与 Hall 定理.canvas]]"
---

# Berge 定理用增广路刻画最大匹配

<!-- bilingual-en:start -->
*Berge's theorem characterises maximum matchings by augmenting paths*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 对有限图中的 matching $M$，
> $$
> M\text{ 是 maximum}\quad\Longleftrightarrow\quad
> \text{不存在相对于 }M\text{ 的 augmenting path}.
> $$
>
> <!-- bilingual-en:start -->
> For a matching $M$ in a finite graph,
> $$
> M\text{ is maximum}\quad\Longleftrightarrow\quad
> \text{there is no augmenting path relative to }M.
> $$
> <!-- bilingual-en:end -->

一个方向立即成立：若存在 augmenting path，[[增广路对称差|翻转]] 后得到更大的 matching，所以当前 $M$ 不可能 maximum。
<!-- bilingual-en:start -->
One direction is immediate: if an augmenting path exists, [[增广路对称差|flipping it]] produces a larger matching, so $M$ cannot be maximum.
<!-- bilingual-en:end -->

反方向解释了这个判据为何完整。若有另一个 matching $M^\star$ 满足 $|M^\star|>|M|$，考察边集为 $M\triangle M^\star$ 的子图。每个顶点的度至多为 $2$；而度为 $2$ 时相接的两条边必分别来自 $M$ 与 $M^\star$。因此每个非平凡连通分量都是 alternating path 或偶长 alternating cycle。cycle 中两种 matching 的边数相等；而总计 $M^\star$ 边更多，因此至少一条 path 中 $M^\star$ 比 $M$ 多一条边。这条 path 以 $M^\star$ 边开始和结束，两端都未被 $M$ 覆盖，正是 $M$ 的 augmenting path。
<!-- bilingual-en:start -->
The reverse direction explains why the criterion is complete. If another matching $M^\star$ satisfies $|M^\star|>|M|$, consider the subgraph with edge set $M\triangle M^\star$. Every vertex has degree at most two, and at a degree-two vertex one incident edge comes from each matching. Hence every nontrivial component is an alternating path or an even alternating cycle. A cycle contains equal numbers of edges from the two matchings. Since $M^\star$ has more edges overall, some path contains one more $M^\star$-edge than $M$-edge. It begins and ends with $M^\star$-edges, so both endpoints are uncovered by $M$ and the path is augmenting for $M$.
<!-- bilingual-en:end -->

定理本身适用于一般有限图，不只适用于二分图。二分图的特殊优势在于，增广路搜索和由搜索得到的 [[交替搜索的增广与缺额证书|Hall bottleneck certificate]] 更容易组织。
<!-- bilingual-en:start -->
The theorem applies to arbitrary finite graphs, not only bipartite ones. Bipartite structure makes augmenting-path search and the resulting [[交替搜索的增广与缺额证书|Hall bottleneck certificate]] easier to organise.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 一个 matching 没有 augmenting path，能否只断言它 maximal？
>
> **答案：** 可以得到更强结论：由 Berge 定理，它是 maximum。

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- MIT OpenCourseWare 18.997，[Lecture 1: Non-Bipartite Matching](https://ocw.mit.edu/courses/18-997-topics-in-combinatorial-optimization-spring-2004/cb34901551f24affa0c147af1cb9151a_co_lec1.pdf)，第 1–2 页：核验 Berge 定理及利用两个 matchings 的 symmetric difference 证明反方向。
  <!-- bilingual-en:start -->
  MIT OpenCourseWare 18.997, [Lecture 1: Non-Bipartite Matching](https://ocw.mit.edu/courses/18-997-topics-in-combinatorial-optimization-spring-2004/cb34901551f24affa0c147af1cb9151a_co_lec1.pdf), pp. 1–2, verifies Berge's theorem and its reverse-direction proof via the symmetric difference of two matchings.
  <!-- bilingual-en:end -->

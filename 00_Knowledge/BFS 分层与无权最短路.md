---
aliases:
  - "BFS 按边数分层并在首次发现时得到无权最短距离"
  - Breadth-first search gives unweighted shortest distances
  - BFS 分层与无权最短路
student_os: knowledge-atom
atom_id: MCS-GRAPH-014
atom_type: theorem
status: source-checked
mastery_state: unassessed
part_of:
  - "[[图的基本结构、路径与遍历.canvas]]"
requires:
  - "[[搜索标记与父森林]]"
  - "[[图距离与可达性]]"
implies:
  - "[[BFS 最短路树的顺序依赖]]"
---

# BFS 按边数分层并在首次发现时得到无权最短距离
<!-- bilingual-en:start -->
*BFS explores by edge-count layers and fixes unweighted shortest distance at first discovery*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 从源点 $s$ 做 breadth-first search（BFS），用 FIFO queue 逐层扩展：
> $$
> L_k=\{v:\operatorname{dist}(s,v)=k\}.
> $$
> 当顶点 $v$ 第一次从 $u$ 被发现时，赋值 $dist[v]=dist[u]+1$；在 unweighted graph 中，这个值就是真实 shortest-path distance。
> <!-- bilingual-en:start -->
> Breadth-first search (BFS) from source $s$ uses a FIFO queue to expand distance layers
> $$
> L_k=\{v:\operatorname{dist}(s,v)=k\}.
> $$
> When $v$ is first discovered from $u$, BFS assigns $dist[v]=dist[u]+1$. In an unweighted graph this is the true shortest-path distance.
> <!-- bilingual-en:end -->

## 分层不变量
<!-- bilingual-en:start -->
*Layer invariant*
<!-- bilingual-en:end -->

FIFO 保证 queue 中待处理顶点的已知距离非递减。处理完 $L_{k-1}$ 前不会开始扩展 $L_k$ 后面的层；从 $u\in L_{k-1}$ 新发现的邻居只能进入 $L_k$。若新发现的 $v$ 实际有更短路径，取那条路径上 $v$ 的前驱 $x$；$x$ 位于更早的层，本应已经处理并更早发现 $v$，与“现在首次发现”矛盾。
<!-- bilingual-en:start -->
FIFO keeps queued vertices in nondecreasing discovered distance. BFS finishes expanding $L_{k-1}$ before vertices beyond $L_k$, and a new neighbor of $u\in L_{k-1}$ enters $L_k$. If newly discovered $v$ had a shorter path, its predecessor $x$ on that path would lie in an earlier layer and would already have discovered $v$, contradicting first discovery now.
<!-- bilingual-en:end -->

这个证明同时说明所有不可达顶点保持 $dist=\infty$。BFS 不只是回答某一个终点：一次搜索返回源点到每个 reachable vertex 的距离，所以它解决 unweighted single-source shortest paths。
<!-- bilingual-en:start -->
The same argument leaves every unreachable vertex at distance infinity. BFS does not answer only one target: one run returns distances from the source to every reachable vertex, solving unweighted single-source shortest paths.
<!-- bilingual-en:end -->

## 权重边界
<!-- bilingual-en:start -->
*Weight boundary*
<!-- bilingual-en:end -->

BFS 最小化的是边数。若各边权重相同，边数与总权重等价；若权重不同，一条权重 $100$ 的直达边会在三条权重 $1$ 的边之前到达终点，但总成本反而更大。此时必须根据权重条件改用 DAG shortest path、Dijkstra 或 Bellman–Ford 等算法。
<!-- bilingual-en:start -->
BFS minimizes edge count. Equal edge weights make edge count proportional to total weight. With unequal weights, a direct edge of weight 100 is discovered before a three-edge route of weights one each, even though its total cost is larger. Use an algorithm appropriate to the weight assumptions, such as DAG shortest paths, Dijkstra, or Bellman–Ford.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 为什么“queue 是 FIFO”不是实现细节，而是 BFS 最短路证明的一部分？
>
> **答案：**FIFO 让较小 distance layer 的顶点先于更大 layer 被扩展。若改成任意取出或 LIFO，较深分支可能先发现顶点，首次发现距离就不再有最短保证。
> <!-- bilingual-en:start -->
> Why is FIFO not merely an implementation detail but part of the BFS shortest-path proof?
>
> **Answer:** FIFO expands every smaller-distance layer before larger ones. Arbitrary removal or LIFO may follow a deep branch first, so first discovery would no longer certify a shortest distance.
> <!-- bilingual-en:end -->

## 来源与核验

- [MIT 6.006 Lecture 9: Breadth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/196a95604877d326c6586e60477b59d4_MIT6_006S20_lec9.pdf)，pp. 3–4：给出 level sets、归纳不变量、distance 与 parent 的正确性及线性时间分析。
  <!-- bilingual-en:start -->
  Lecture 9 gives the level sets, inductive invariant, correctness of distances and parents, and the linear-time analysis.
  <!-- bilingual-en:end -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=437|MIT Mathematics for Computer Science, Section 10.3.1]]：核验 shortest-path distance 是最早出现非零 walk count 的长度，且 simple path 最长为 $|V|-1$。
  <!-- bilingual-en:start -->
  Section 10.3.1 verifies the relationship between shortest distance and the first positive walk-counting power.
  <!-- bilingual-en:end -->

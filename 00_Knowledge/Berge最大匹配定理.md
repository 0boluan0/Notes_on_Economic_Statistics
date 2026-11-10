---
aliases:
  - Berge 定理
  - Berge's theorem
  - Augmenting-path criterion for maximum matching
student_os: knowledge-atom
atom_id: MCS-MATCH-007
atom_type: theorem
status: source-checked
requires:
  - "[[二分图]]"
  - "[[匹配定义]]"
  - "[[极大与最大匹配]]"
  - "[[增广路定义]]"
  - "[[增广路对称差]]"
part_of:
  - "[[二分图匹配与 Hall 定理.canvas]]"
---

<!-- bilingual-en:start -->
# Berge's theorem
<!-- bilingual-en:end -->
*Berge 最大匹配定理*

<!-- bilingual-en:start -->
A [[极大与最大匹配|maximum matching]] selects as many nonconflicting pairs as the graph permits. How can we know that a current [[匹配定义|matching]] cannot be improved, even by replacing several old pairs? **Berge’s theorem** gives an exact answer: there is no [[增广路定义|augmenting path]] for that matching. Such a path is a simple chain of alternating unselected and selected edges, of positive length, whose two distinct endpoints are unmatched in the whole graph.
<!-- bilingual-en:end -->
[[极大与最大匹配|最大匹配]]选择图中尽可能多的互不冲突配对。怎样知道当前[[匹配定义|匹配]]已经无法改进，连撤换若干旧配对也不行？**Berge 定理**给出准确答案：相对于当前匹配，不存在[[增广路定义|增广路]]。这种路是正长度简单路径，边在未选与已选之间交替，两个不同端点在全图中均未匹配。

<!-- bilingual-en:start -->
## The four-vertex improvement
<!-- bilingual-en:end -->
*四个顶点上的改进*

<!-- bilingual-en:start -->
Take four distinct vertices $b,1,a,2$ and just the three edges of $b-1-a-2$. With $M=\{a-1\}$, no additional edge can be inserted without conflict: $M$ is maximal. Yet removing $a-1$ and adding $b-1,a-2$ gives two edges. Thus checking only whether one edge can be added misses an improvement. The alternating chain records exactly which old choice must be undone.
<!-- bilingual-en:end -->
取四个不同顶点 $b,1,a,2$，只有路径 $b-1-a-2$ 的三条边。令 $M=\{a-1\}$，此时不能直接加入任何一条边而不冲突，所以 $M$ 是极大匹配。然而撤下 $a-1$，改选 $b-1,a-2$，就能得到两条边。只检查能否直接加边，会漏掉这种改进；交替链准确记录了需要撤回的旧选择。

<!-- bilingual-en:start -->
## An exact criterion
<!-- bilingual-en:end -->
*准确的必要充分判据*

<!-- bilingual-en:start -->
Let $G=(V,E)$ be a finite [[图类型的建模边界|simple undirected graph]], and let $M\subseteq E$ be a matching. Maximum means that every other matching $N$ satisfies $|N|\le |M|$; it does not require $N$ to contain $M$.
<!-- bilingual-en:end -->
设 $G=(V,E)$ 是有限[[图类型的建模边界|简单无向图]]，$M\subseteq E$ 是匹配。最大是指任何其他匹配 $N$ 都满足 $|N|\le |M|$，比较时不要求 $N$ 包含 $M$。

> [!important] Berge’s theorem / Berge 定理
> <!-- bilingual-en:start -->
> For a matching $M$ in a finite graph,
> $$
> M\text{ is maximum}\quad\Longleftrightarrow\quad
> \text{there is no augmenting path relative to }M.
> $$
> <!-- bilingual-en:end -->
> 对有限图中的 matching $M$，
> $$
> M\text{ 是 maximum}\quad\Longleftrightarrow\quad
> \text{不存在相对于 }M\text{ 的 augmenting path}.
> $$

<!-- bilingual-en:start -->
One direction is immediate: if an augmenting path exists, [[增广路对称差|flipping it]] produces a larger matching, so $M$ cannot be maximum.
<!-- bilingual-en:end -->
一个方向立即成立：若存在 augmenting path，[[增广路对称差|翻转]] 后得到更大的 matching，所以当前 $M$ 不可能 maximum。

<!-- bilingual-en:start -->
## Compare the current matching with a larger one
<!-- bilingual-en:end -->
*把当前匹配与更大匹配放在一起比较*

<!-- bilingual-en:start -->
For the converse, suppose $M$ is not maximum. Choose a matching $M^\star$ with $|M^\star|>|M|$. We need not know how to find it: the definition of nonmaximum guarantees its existence. Keep only the edges on which the two matchings disagree:
<!-- bilingual-en:end -->
反过来，假设 $M$ 不是最大匹配，就存在匹配 $M^\star$ 满足 $|M^\star|>|M|$。证明不要求先知道怎样找到它；不是最大的定义已经保证它存在。只保留两个匹配意见不同的边：

$$
D=M\mathbin{\triangle}M^\star
=(M\setminus M^\star)\cup(M^\star\setminus M).
$$

<!-- bilingual-en:start -->
This is a symmetric difference: common edges disappear. Color edges in $M\setminus M^\star$ red and edges in $M^\star\setminus M$ blue. At a vertex there is at most one red edge and at most one blue edge, because each original set is a matching. Thus its [[度数、入度与出度|degree]] (number of incident edges) in the graph $(V,D)$ is at most two; if two edges meet, they have different colors.
<!-- bilingual-en:end -->
这就是对称差：公共边被去掉。把 $M\setminus M^\star$ 中的边涂红，把 $M^\star\setminus M$ 中的边涂蓝。每个顶点至多接一条红边、至多接一条蓝边，因为原来两个集合各自都是匹配。因此在子图 $(V,D)$ 中，每个顶点的[[度数、入度与出度|度]]（接触的边数）至多为二；若有两条边相接，它们的颜色必不同。

<!-- bilingual-en:start -->
A finite [[无向图连通分量|connected component]] with an edge and maximum degree two is a simple path or a cycle. To see why, start at a degree-one vertex if one exists and keep following the sole unused continuation. No branching is possible; the finite chain ends and exhausts that component. If every vertex has degree two, following edges eventually closes a cycle, which exhausts the component because no vertex has an extra edge leaving it. Isolated vertices contribute no edges. The red and blue colors alternate on every path and cycle; a cycle must therefore have even length.
<!-- bilingual-en:end -->
有限[[无向图连通分量|连通分量]]若含边且每个顶点的度至多为二，就只能是一条简单路径或一个环。若存在度为一的顶点，从它出发不断沿唯一尚未使用的延续边前进：无法分叉，有限链最终结束，并用尽这个分量。若每个顶点的度都是二，沿边前进最终闭合成环；环上顶点没有多余边通向外面，所以它就是整个分量。孤立点不贡献边。每条路径与环上的红蓝颜色交替，因此环的长度一定为偶数。

<!-- bilingual-en:start -->
An alternating cycle has equally many red and blue edges. On an alternating path, blue minus red is $-1$, $0$, or $1$. Common edges contribute equally to the original matchings, so the sum of these component differences is $|M^\star|-|M|>0$. Consequently at least one path has one more blue edge. It begins and ends with blue edges.
<!-- bilingual-en:end -->
交替环的红蓝边数相等。交替路径上，蓝边数减红边数只能是 $-1$、$0$ 或 $1$。公共边对两个原匹配的计数贡献相同，因此各分量差值之和就是 $|M^\star|-|M|>0$。于是至少有一条路径多一条蓝边；它以蓝边开始，也以蓝边结束。

<!-- bilingual-en:start -->
## Why its endpoints are unmatched in the whole graph
<!-- bilingual-en:end -->
*为什么两端在全图中都未匹配*

<!-- bilingual-en:start -->
Let $v$ be either endpoint of this path. It has a blue edge from $M^\star\setminus M$. Suppose an edge of $M$ were incident to $v$. If that edge also belonged to $M^\star$, then $M^\star$ would have two distinct edges at $v$, impossible. If it did not belong to $M^\star$, it would be a red edge in $D$ at $v$, so $v$ would not be an endpoint of this component. Both possibilities fail. Thus $v$ has no $M$-edge anywhere in $G$.
<!-- bilingual-en:end -->
取这条路径的任一端点 $v$，它接着一条来自 $M^\star\setminus M$ 的蓝边。假设另有一条 $M$ 中的边接触 $v$：若它也属于 $M^\star$，则 $M^\star$ 在 $v$ 处接着两条不同的边，违反匹配规则；若它不属于 $M^\star$，它就是 $D$ 中接触 $v$ 的红边，于是 $v$ 就不是该分量的端点。两种可能都被排除，所以 $v$ 在全图中没有任何 $M$ 边。

<!-- bilingual-en:start -->
The selected component is therefore a positive-length simple alternating path with two distinct globally unmatched endpoints: an $M$-augmenting path. This proves the converse by contrapositive. The following compact account summarizes the same argument; the endpoint step is justified by the preceding paragraph.
<!-- bilingual-en:end -->
因此，所选分量是正长度简单交替路，两个不同端点在全图中都未被 $M$ 匹配，正是一条 $M$-增广路。这就通过逆否命题证明了反方向。下面把同一论证压缩成一段；其中端点未匹配的步骤，由上一段完整说明。

<!-- bilingual-en:start -->
The reverse direction explains why the criterion is complete. If another matching $M^\star$ satisfies $|M^\star|>|M|$, consider the subgraph with edge set $M\triangle M^\star$. Every vertex has degree at most two, and at a degree-two vertex one incident edge comes from each matching. Hence every nontrivial component is an alternating path or an even alternating cycle. A cycle contains equal numbers of edges from the two matchings. Since $M^\star$ has more edges overall, some path contains one more $M^\star$-edge than $M$-edge. It begins and ends with $M^\star$-edges, so both endpoints are uncovered by $M$ and the path is augmenting for $M$.
<!-- bilingual-en:end -->
反方向解释了这个判据为何完整。若有另一个 matching $M^\star$ 满足 $|M^\star|>|M|$，考察边集为 $M\triangle M^\star$ 的子图。每个顶点的度至多为 $2$；而度为 $2$ 时相接的两条边必分别来自 $M$ 与 $M^\star$。因此每个非平凡连通分量都是 alternating path 或偶长 alternating cycle。cycle 中两种 matching 的边数相等；而总计 $M^\star$ 边更多，因此至少一条 path 中 $M^\star$ 比 $M$ 多一条边。这条 path 以 $M^\star$ 边开始和结束，两端都未被 $M$ 覆盖，正是 $M$ 的 augmenting path。

<!-- bilingual-en:start -->
## What the theorem does and does not require
<!-- bilingual-en:end -->
*定理的适用范围*

<!-- bilingual-en:start -->
The theorem applies to arbitrary finite graphs, not only [[二分图|bipartite]] ones. Bipartite structure makes augmenting-path search and the resulting [[交替搜索的增广与缺额证书|Hall bottleneck certificate]] easier to organise.
<!-- bilingual-en:end -->
定理本身适用于一般有限图，不只适用于[[二分图]]。二分图的特殊优势在于，增广路搜索和由搜索得到的 [[交替搜索的增广与缺额证书|Hall bottleneck certificate]] 更容易组织。

<!-- bilingual-en:start -->
For example, a triangle is not bipartite, but any one of its edges is a maximum matching. There is only one unmatched vertex, so no augmenting path can have two unmatched endpoints. Maximum need not mean perfect. For an edgeless graph, including the empty graph, the empty matching is maximum and there is no augmenting path. A perfect matching is also maximum, since each selected edge uses two distinct vertices.
<!-- bilingual-en:end -->
例如三角形不是二分图，但任选其中一条边就是最大匹配。此时只有一个未匹配顶点，无法组成具有两个未匹配端点的增广路。最大未必完美。对无边图，包括空图，空匹配是最大匹配，也不存在增广路。完美匹配同样必为最大匹配，因为每条选中边都使用两个不同顶点。

<!-- bilingual-en:start -->
Starting from any matching, each successful flip increases its size by one; in a finite graph this can happen at most $\lfloor |V|/2\rfloor-|M|$ times. Berge’s theorem certifies the result when **no augmenting path exists**. A search that merely fails to find one, without exploring everything required by a correct search method, does not establish that premise. The theorem concerns cardinality, not total edge weight, and does not by itself specify an efficient search algorithm for general graphs.
<!-- bilingual-en:end -->
从任意匹配出发，每次成功翻转都使边数增加一；有限图中至多还能增加 $\lfloor |V|/2\rfloor-|M|$ 次。当**不存在任何增广路**时，Berge 定理确认所得匹配最大。某次搜索没有找到路径，若并未按正确方法检查所需范围，就不能证明这个前提。定理优化的是边数，不是总边权，也没有自行给出一般图中的高效搜索算法。

> [!question]- Maximum or merely maximal? / 回忆与辨别
> <!-- bilingual-en:start -->
> If a matching has no augmenting path, can we conclude only maximality? Berge’s theorem gives the stronger conclusion: it is maximum.
> <!-- bilingual-en:end -->
> 一个 matching 没有 augmenting path，能否只断言它 maximal？
>
> **答案：** 可以得到更强结论：由 Berge 定理，它是 maximum。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- MIT OpenCourseWare 18.997, [Lecture1: Nonbipartite Matching](https://ocw.mit.edu/courses/18-997-topics-in-combinatorial-optimization-spring-2004/cb34901551f24affa0c147af1cb9151a_co_lec1.pdf), PDF2–3, Theorem4: Berge’s criterion for general graphs and comparison of two matchings by alternating components. The proof above explicitly removes common edges, counts component differences, and checks endpoints in the whole graph.
- Diestel, [Graph Theory, Chapter2](https://www.math.uni-hamburg.de/home/diestel/books/graph.theory/preview/Ch2.pdf), PDF2 / printed38, §2.1: flipping an augmenting path and the improvement criterion in the bipartite setting; the general-graph scope comes from MIT’s theorem.
<!-- bilingual-en:end -->
- MIT OpenCourseWare 18.997，[Lecture1: Nonbipartite Matching](https://ocw.mit.edu/courses/18-997-topics-in-combinatorial-optimization-spring-2004/cb34901551f24affa0c147af1cb9151a_co_lec1.pdf)，PDF2–3，Theorem4：一般图的 Berge 判据及交替分量比较思路。上文明确去掉公共边、逐分量计数，并核查端点在全图中的匹配状态。
- Diestel，[Graph Theory 第2章](https://www.math.uni-hamburg.de/home/diestel/books/graph.theory/preview/Ch2.pdf)，PDF2／印38页，§2.1：二分图中的增广翻转与改进判据；一般图范围由 MIT 定理支持。

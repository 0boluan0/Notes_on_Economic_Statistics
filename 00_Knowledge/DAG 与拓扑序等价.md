---
aliases:
- DAG and topological order equivalence
- 有向无环与拓扑序的等价性
student_os: knowledge-atom
atom_id: MCS-GRAPH-033
atom_set: acyclic-graphs-trees-dags
atom_type: theorem
status: source-checked
requires:
- '[[Tree 与 DAG 的区别]]'
- '[[度数、入度与出度]]'
- '[[游走路径与环]]'
related:
- '[[有向 DFS 的环证书]]'
leads_to:
- '[[Kahn 拓扑排序与环证书]]'
- '[[DFS 逆完成序拓扑排序]]'
- '[[拓扑序的线性扩张]]'
part_of:
- '[[无环图：树、生成树、DAG 与拓扑排序.canvas]]'
---

<!-- bilingual-en:start -->
# DAGs and topological orders
<!-- bilingual-en:end -->

*DAG 与拓扑序的等价性*

<!-- bilingual-en:start -->
Suppose each arrow means “do this task before that one.” A **[[拓扑序|topological order]]** puts every task into one list while respecting every arrow. The key existence question is whether such a list can be made at all. For a finite directed graph, it can be made exactly when the graph has no directed cycle: a circular chain of prerequisites is the obstruction.
<!-- bilingual-en:end -->

设每条箭头表示“先做这项任务，再做那项”。**[[拓扑序]]**把全部任务排成一列，同时遵守每条箭头。这里的核心问题是：这样的列表到底是否存在？对有限有向图，答案恰好等价于图中没有有向环；循环先修关系正是障碍。

<!-- bilingual-en:start -->
Let $G=(V,E)$ be a finite [[图类型的建模边界|directed graph]], with vertex set $V$ and edge set $E$. A **[[有向无环图|directed acyclic graph]]**, abbreviated **DAG**, has no directed [[游走路径与环|cycle]]. Write a topological order as $v_1,\ldots,v_n$, where $n=|V|$ counts the vertices, each vertex appears exactly once, and every edge $v_i\to v_j$ has $i<j$. Thus “before” is strict. The zero-length path at a vertex does not form a directed cycle.
<!-- bilingual-en:end -->

令 $G=(V,E)$ 为有限[[图类型的建模边界|有向图]]，$V$ 是顶点集，$E$ 是边集。**[[有向无环图]]**（directed acyclic graph，简称 **DAG**）没有有向[[游走路径与环|环]]。拓扑序写成 $v_1,\ldots,v_n$，其中 $n=|V|$ 是顶点数，每个顶点恰好出现一次，且每条边 $v_i\to v_j$ 都满足 $i<j$，所以“在前”是严格的。顶点处的零长度路径不构成有向环。

> [!example] Two prerequisites and a circular obstruction · 两项先修与循环障碍
> <!-- bilingual-en:start -->
> With $a\to c$ and $b\to c$, both $a,b,c$ and $b,a,c$ work: the two prerequisites must precede $c$, but the arrows do not order $a$ against $b$. If we also add $c\to a$, the pair $a\to c\to a$ would demand that $a$ occur both before and after $c$. No full ordering can satisfy it.
> <!-- bilingual-en:end -->
>
> 只有 $a\to c$ 与 $b\to c$ 时，$a,b,c$ 和 $b,a,c$ 都合法：两项先修必须在 $c$ 前面，但箭头没有规定 $a,b$ 的相对次序。若再加入 $c\to a$，则 $a\to c\to a$ 同时要求 $a$ 在 $c$ 之前与之后，不可能存在满足全部约束的排列。

<!-- bilingual-en:start -->
## The local test must hold in every remaining subgraph
<!-- bilingual-en:end -->

*局部条件必须对每个剩余子图成立*

<!-- bilingual-en:start -->
Choose a vertex subset $S\subseteq V$. The **induced subgraph** on $S$ keeps exactly those vertices and every original edge whose two endpoints lie in $S$. Within that subgraph, a **source** has [[度数、入度与出度|in-degree]] zero: no retained edge enters it. A **sink** has out-degree zero: no retained edge leaves it. These counts refer to the subgraph, not to the unchanged original graph.
<!-- bilingual-en:end -->

选取顶点子集 $S\subseteq V$。$S$ 上的**诱导子图**保留这些顶点，以及原图中两端都位于 $S$ 内的全部边。在该子图中，**源点**的[[度数、入度与出度|入度]]为零，即没有保留下来的边进入它；**汇点**的出度为零，即没有保留下来的边离开它。这里统计的是当前子图，不是一直使用原图的度数。

> [!theorem] Four equivalent conditions · 四个等价条件
> <!-- bilingual-en:start -->
> For a finite directed graph, the following are equivalent.
>
> - The graph is a DAG.
> - It has a topological order of all its vertices.
> - Every **nonempty** induced subgraph has a source.
> - Every **nonempty** induced subgraph has a sink.
> <!-- bilingual-en:end -->
>
> 对有限有向图，以下四项等价。
>
> - 图是 DAG。
> - 全部顶点存在拓扑序。
> - 每个**非空**诱导子图都有源点。
> - 每个**非空**诱导子图都有汇点。

<!-- bilingual-en:start -->
## Why a valid order rules out a cycle
<!-- bilingual-en:end -->

*合法顺序为什么排除环*

<!-- bilingual-en:start -->
Assign each vertex its position in the list. Along a directed edge the position increases strictly. Along a directed cycle it would therefore keep increasing and return to its starting position, which is impossible. This also rules out a self-loop: its endpoint would have to precede itself. The argument proves the implication from an order to acyclicity without running any search algorithm.
<!-- bilingual-en:end -->

给每个顶点标上它在列表中的位置。每沿一条有向边，位置都严格增大；若存在有向环，位置就必须不断增大后又回到起点，矛盾。自环也被排除，因为它要求顶点排在自己之前。这直接证明“存在拓扑序则无环”，无需先执行搜索算法。

<!-- bilingual-en:start -->
## Why acyclicity supplies a next vertex
<!-- bilingual-en:end -->

*无环为什么保证下一步可选*

<!-- bilingual-en:start -->
First take a **nonempty finite** digraph with no source. Choose a vertex $w_0$. Since an edge enters it, choose $w_1\to w_0$; then choose $w_2\to w_1$, and continue. There are only finitely many vertices, so the sequence eventually repeats a vertex. Take the segment between the two occurrences of that first repeated vertex. Read that segment in the opposite order: its edges now run forward and form a directed cycle. Consequently, a nonempty finite DAG must have a source.
<!-- bilingual-en:end -->

先考虑一张**非空有限**有向图，假设它没有源点。选一个顶点 $w_0$，因为存在进入它的边，可选 $w_1\to w_0$；再选 $w_2\to w_1$，继续如此。顶点只有有限个，因此序列最终重复。取这个首次重复顶点两次出现之间的闭段，并反向读这段序列，边就沿正向连接成有向环。因此，非空有限 DAG 必有源点。

<!-- bilingual-en:start -->
Every induced subgraph of a DAG is still acyclic, since deleting vertices cannot create a cycle. The source argument therefore applies to every nonempty induced subgraph. Reversing all arrows preserves whether a directed cycle exists and exchanges sources with sinks, so the same statement holds for sinks.
<!-- bilingual-en:end -->

DAG 的每个诱导子图仍然无环，因为删除顶点不会创造环。因此，源点论证适用于每个非空诱导子图。把全部箭头反向，是否存在有向环不变，而源点与汇点互换，所以同样得到汇点结论。

<!-- bilingual-en:start -->
## Build the order by removing sources
<!-- bilingual-en:end -->

*逐次删除源点，构造整个顺序*

<!-- bilingual-en:start -->
Now assume every nonempty induced subgraph has a source. Choose one, put it next in the output, and remove it and its incident edges. Repeat on the remaining induced subgraph. Each step removes one vertex, so after $n$ steps every vertex has appeared once. For an edge $u\to v$, vertex $v$ could not have been removed while $u$ was still present: that edge would give $v$ positive in-degree. Hence $u$ precedes $v$, and the output is a topological order. This is the structural reason behind [[Kahn 拓扑排序与环证书|Kahn's algorithm]].
<!-- bilingual-en:end -->

现在假设每个非空诱导子图都有源点。选一个源点作为下一个输出，删除它及其关联边，再对剩余诱导子图重复。每步减少一个顶点，所以 $n$ 步后全部顶点恰好输出一次。对边 $u\to v$，只要 $u$ 还在，$v$ 就因这条边而有正入度，不能先被删除。因此 $u$ 必在 $v$ 前，输出就是拓扑序。这是 [[Kahn 拓扑排序与环证书|Kahn 算法]]背后的结构保证。

<!-- bilingual-en:start -->
If the assumption instead gives a sink in every nonempty induced subgraph, place a chosen sink in the **last still-empty position**, delete it, and fill the list from right to left. Equivalently, apply the source construction to the reverse graph and reverse its output. This proves the sink condition also supplies a topological order, completing all four implications.
<!-- bilingual-en:end -->

如果已知的是每个非空诱导子图都有汇点，就把选中的汇点放进**尚未填写的最后一个位置**，删除它，再从右向左填列表；等价地，对反图用源点法，再反转输出。这证明汇点条件也能给出拓扑序，四个条件于是全部等价。

> [!warning] One source in the original graph is not enough · 原图里有一个源点还不够
> <!-- bilingual-en:start -->
> In $a\to b$, $b\to c$, $c\to b$, vertex $a$ is a source, yet $b,c$ form a cycle. Deleting $a$ exposes a nonempty remainder with no source. The theorem requires a source in **every nonempty induced subgraph**, not merely one in the original graph. In a DAG, a vertex with positive original in-degree may become a source after its predecessors are removed.
> <!-- bilingual-en:end -->
>
> 在 $a\to b$、$b\to c$、$c\to b$ 中，$a$ 是源点，但 $b,c$ 形成环。删掉 $a$ 后，就留下没有源点的非空子图。定理要求**每个非空诱导子图**都有源点，而不只是原图有一个。对 DAG 而言，原来入度为正的顶点，也可能在前驱删除后成为新源点。

<!-- bilingual-en:start -->
If empty graphs are allowed, the empty list is their topological order; the two conditions about nonempty induced subgraphs are vacuously true. Isolated vertices must still appear in a nonempty graph's order. Parallel copies of an arrow impose the same ordering constraint, while a self-loop forbids an order. Nothing here asserts a unique order or a [[拓扑序的线性扩张|resource-optimal schedule]]. Another construction, [[DFS 逆完成序拓扑排序|reverse DFS finishing order]], proves the same edge-order requirement by a different route.
<!-- bilingual-en:end -->

若允许空图，则空列表就是它的拓扑序；关于非空诱导子图的两项要求都真空成立。非空图中的孤立顶点仍须出现在排列里。箭头的平行副本施加相同的先后约束，而自环使拓扑序不存在。定理没有声称顺序唯一，也没有保证[[拓扑序的线性扩张|资源调度最优]]。另一种构造[[DFS 逆完成序拓扑排序|DFS 逆完成序]]以不同方式证明每条边的先后要求。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=439|MIT Mathematics for Computer Science (2018), §10.5]], PDF 439 and 441–443: DAG definition 10.5.1, topological list definition 10.5.2, minimal/source deletion and theorem 10.5.4. The finite no-source cycle argument, induced-subgraph conditions and empty-graph extension are made explicit above.
- [MIT 6.006 Lecture 10: Depth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf), PDF4: edge-position definition and the DAG/order equivalence.
- [[01_Math/07-Mathematics for Computer Science/02_Structures.md#17.1 DAG 与拓扑排序|Permanent Session 17.1]]: prerequisite interpretation, source deletion and equivalence context.
<!-- bilingual-en:end -->

- MIT《Mathematics for Computer Science》（2018）§10.5，PDF439及441–443：DAG 定义10.5.1、拓扑列表定义10.5.2、minimal/source 删除构造与定理10.5.4。正文展开有限无源图产生环的证明、诱导子图条件，并明确空图扩展。
- MIT 6.006 Lecture 10，PDF4：逐边位置定义以及 DAG 与拓扑序等价。
- 永久 Session17.1：先修关系、删除源点构造及等价条件语境。

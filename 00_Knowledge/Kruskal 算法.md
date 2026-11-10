---
aliases:
  - Kruskal algorithm for minimum spanning trees
  - Kruskal 最小生成树算法
student_os: knowledge-atom
atom_id: MCS-GRAPH-031
atom_set: acyclic-graphs-trees-dags
atom_type: algorithm
status: source-checked
requires:
  - "[[Forest 的边数公式]]"
  - "[[MST 割性质]]"
related:
  - "[[无向 DFS 的父边例外]]"
  - "[[Prim 算法]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

<!-- bilingual-en:start -->
# Kruskal’s algorithm
<!-- bilingual-en:end -->

*Kruskal 算法*

<!-- bilingual-en:start -->
Kruskal’s algorithm builds a cheap network by considering edges from cheapest to most expensive. It keeps an edge only when that edge connects two pieces that are still separate. This second test matters: cheap edges can form a cycle before every vertex has been connected. The goal is a [[最小生成树的定义与唯一性|minimum spanning tree (MST)]], which connects all vertices without cycles and minimizes the sum of the selected edge weights.
<!-- bilingual-en:end -->

Kruskal 算法按边权从小到大考虑连线，但只有当一条边能连接当前仍分开的两部分时，才保留它。第二项检查不可省略：便宜的边也可能先围成环，却没有连到全部顶点。目标是[[最小生成树的定义与唯一性|最小生成树（MST）]]，即无环地连通全部顶点，并使所选边的权重总和最小。

<!-- bilingual-en:start -->
Let $G=(V,E)$ be a finite, nonempty, connected [[图类型的建模边界|undirected graph]]. Here $V$ is its vertex set, $E$ its edge set, and each edge $e$ has a finite real weight $w(e)$. Zero, negative and equal weights are allowed. Parallel edges, if present, retain their separate identities. A self-loop cannot belong to a tree and will fail the component test below. The disconnected case is treated at the end.
<!-- bilingual-en:end -->

设 $G=(V,E)$ 是有限、非空、连通的[[图类型的建模边界|无向图]]。$V$ 是顶点集，$E$ 是边集，每条边 $e$ 有有限实数权重 $w(e)$；允许零权、负权和并列权重。若有平行边，必须保留各自身份。自环不能属于树，也不会通过下文的分量检查。最后再讨论不连通图。

<!-- bilingual-en:start -->
## Keep a forest while scanning all edges
<!-- bilingual-en:end -->

*扫描全部边，同时保持森林*

<!-- bilingual-en:start -->
Start with every vertex present and no selected edges: $F=\varnothing$. The graph $(V,F)$ is a [[森林|forest]], an undirected graph with no cycle. Initially every vertex is its own [[无向图连通分量|connected component]]: a component is a maximal group of vertices joined by paths using the currently selected edges. These are components of $(V,F)$, not components of the original connected input.
<!-- bilingual-en:end -->

保留全部顶点，所选边集从空集 $F=\varnothing$ 开始。$(V,F)$ 是一片[[森林]]，即没有环的无向图。起初每个顶点各自构成一个[[无向图连通分量|连通分量]]；这里的分量是沿当前所选边能够互达的极大顶点组。它们属于 $(V,F)$，不是原来那张连通输入图的分量。

<!-- bilingual-en:start -->
Sort all edges in **nondecreasing** weight order: smaller weights first, with equal weights in any order. When an edge $e$ with endpoints $u,v$ is reached, accept it into $F$ exactly when $u,v$ lie in different current components. Merge those two components after accepting it. Otherwise skip it. The test uses paths already built from $F$, not whether $u$ and $v$ have ever appeared in earlier edges.
<!-- bilingual-en:end -->

将全部边按权重**不减**排序：小权在前，相同权重可以任意排列。扫描到端点为 $u,v$ 的边 $e$ 时，当且仅当两端位于当前不同分量，才把它加入 $F$，并合并这两个分量；否则跳过。检查的是所选边 $F$ 已经形成的路径，不是端点是否曾在较早的边中出现过。

> [!example] A cheap edge can still be rejected · 便宜的边也可能被跳过
> <!-- bilingual-en:start -->
> Use four vertices $a,b,c,d$ and exactly these six undirected edges: $w(ab)=1$, $w(bc)=2$, $w(ac)=3$, $w(cd)=4$, $w(bd)=5$, $w(ad)=8$. Here $ab$ means the edge joining $a$ and $b$. Accept $ab$, then $bc$: the components become $\{a,b,c\}$ and $\{d\}$. Reject $ac$ even though its weight is only $3$, because $a-b-c$ already connects its endpoints. Accept $cd$ to reach $d$. The resulting tree is $\{ab,bc,cd\}$ and has total weight $1+2+4=7$.
> <!-- bilingual-en:end -->
>
> 取四个顶点 $a,b,c,d$，只有以下六条无向边：$w(ab)=1$、$w(bc)=2$、$w(ac)=3$、$w(cd)=4$、$w(bd)=5$、$w(ad)=8$；$ab$ 表示连接 $a,b$ 的边。先接受 $ab$、$bc$，分量变成 $\{a,b,c\}$ 与 $\{d\}$。虽然 $ac$ 的权重只有 $3$，仍须跳过，因为已有路径 $a-b-c$ 连接其两端。再接受 $cd$，到达 $d$。结果为 $\{ab,bc,cd\}$，总权重 $1+2+4=7$。

<!-- bilingual-en:start -->
An accepted edge joins two different trees, so it cannot close a cycle. An unselected edge whose endpoints are already in the same tree would close a cycle with the unique existing path; two distinct parallel edges can make a two-edge cycle. Thus $F$ stays a forest. Each acceptance reduces its component count by one. By the [[Forest 的边数公式|forest edge-count formula]], reaching one component means accepting exactly $n-1$ edges, where $n=|V|$ is the number of vertices.
<!-- bilingual-en:end -->

接受一条边是在连接两棵不同的树，不会围成环。若一条尚未选入的边两端已在同一棵树中，它与原有的唯一路径就会成环；两条不同的平行边可以形成二边环。因此 $F$ 始终是森林。每次接受一条边，分量数减少一。由[[Forest 的边数公式|森林边数公式]]，当 $n=|V|$ 个顶点连成一个分量时，恰好接受了 $n-1$ 条边。

<!-- bilingual-en:start -->
## Why the greedy choices are globally safe
<!-- bilingual-en:end -->

*为什么局部选择保留全局最优解*

<!-- bilingual-en:start -->
The weight argument needs a second invariant: **all edges in $F$ together are contained in some MST**. It is not enough for each selected edge separately to belong to some possibly different MST. Initially the empty set has this property because a finite connected graph has at least one MST.
<!-- bilingual-en:end -->

权重最优性还需要第二个不变量：**整个已选边集 $F$ 共同包含于某一棵 MST**。仅仅每条边分别属于某棵可能不同的 MST，并不足够。初始空集满足这个条件，因为有限连通图至少有一棵 MST。

<!-- bilingual-en:start -->
Suppose the next accepted edge is $e=uv$, and let $S$ be the current component containing $u$. A [[图割|cut]] separates $S$ from its complement $V\setminus S$; an edge crosses it when its endpoints lie on opposite sides. No edge of $F$ crosses this cut. If a strictly lighter crossing edge $f$ existed, it would have been scanned earlier. Its endpoints could not have been connected then: components only merge, and the endpoints are still separated now. Hence $f$ would have been accepted, which would already join its two sides and contradict the current component $S$. So $e$ is a minimum-weight crossing edge; ties do not affect the argument.
<!-- bilingual-en:end -->

设下一条接受的边为 $e=uv$，令 $S$ 是当前包含 $u$ 的分量。[[图割|割]]把 $S$ 与补集 $V\setminus S$ 分开；两端分处两侧的边称为跨割边。$F$ 没有边跨这个割。若存在严格更轻的跨割边 $f$，它应当更早被扫描；当时其两端也不可能已连通，因为分量只会合并，而两端现在仍分开。所以当时必已接受 $f$，它就会把两侧连接起来，与当前分量 $S$ 矛盾。因此 $e$ 是跨割最轻边，并列不影响论证。

<!-- bilingual-en:start -->
The [[MST 割性质|safe-edge form of the cut property]] now applies: the current set lies in one MST, the cut crosses none of its edges, and $e$ is light across it. Adding $e$ preserves extendability to an MST. Finally, scanning every edge cannot leave two selected components within a connected input: an original path between them would contain an edge crossing the final components, and that edge could not have been skipped. Thus the result is spanning. A spanning tree contained in another spanning tree must equal it, so the final $F$ is an MST.
<!-- bilingual-en:end -->

现在可用[[MST 割性质|割性质的安全边形式]]：当前边集包含于某棵 MST，所取割不跨当前边集，而 $e$ 是跨割最轻边，故加上 $e$ 后仍可扩展成某棵 MST。最后，连通输入扫描完全部边后不可能还剩两个所选分量：原图连接它们的路径上必有一条跨最终分量的边，而这条边不可能被跳过。因此结果覆盖并连通全部顶点。生成树若包含于另一棵生成树，只能与它相同，最终 $F$ 就是 MST。

<!-- bilingual-en:start -->
## Components, ties and implementation
<!-- bilingual-en:end -->

*分量、并列与实现*

<!-- bilingual-en:start -->
For a disconnected input, scanning all edges gives one MST in each original connected component, including isolated vertices as one-vertex trees. This is a **minimum spanning forest**: the requirement is to span each original component, not to choose an arbitrary forest with the smallest weight. Componentwise optimality follows by applying the same proof separately; the choices in different components cannot help or obstruct one another. An empty graph gives an empty forest under the empty-input convention, not a nonempty tree.
<!-- bilingual-en:end -->

输入不连通时，扫描全部边会在原图每个连通分量中得到一棵 MST；孤立顶点算单点树。这叫**最小生成森林**：要求覆盖原图各个分量，并不是在任意森林中直接挑权重最小者。对每个原分量分别应用同一证明即可得到最优性，不同分量的选择互不干涉。允许空输入时得到空森林，并非非空树。

<!-- bilingual-en:start -->
A disjoint-set structure, also called union–find, can maintain the partition: a “find” operation returns a component representative, and “union” merges two components. Its internal parent pointers are bookkeeping and need not be edges of the selected graph. The proof depends on the partition test, not on this implementation. Equal weights can lead to different MSTs, but need not; negative weights simply appear earlier in the sorted list. A singleton needs no accepted edges. To grow only one component from a chosen start instead, compare [[Prim 算法|Prim’s algorithm]].
<!-- bilingual-en:end -->

可以用不相交集合结构（并查集，union–find）维护分组：“find”查询所在分量的代表，“union”合并两个分量。数据结构内部的父指针只是记账，不一定是所选图中的边。正确性依赖分量检查，而不依赖这一特定实现。并列权重可能产生不同 MST，但并非必然；负权只需排在排序的较前位置。单点图无需选边。若想从一个起点只生长一棵树，可比较[[Prim 算法]]。

> [!question]- Why not accept every cheap edge? · 为什么不把便宜边全选上？
> <!-- bilingual-en:start -->
> Because the output must be a tree. In the example, accepting all of $ab,bc,ac$ creates a cycle before $d$ is reached. Even a negative edge is not eligible if it closes a cycle; minimizing weight is subject to the spanning-tree constraint.
> <!-- bilingual-en:end -->
>
> 因为输出必须是一棵树。例中若把 $ab,bc,ac$ 全部选上，会在连到 $d$ 之前先成环。即使某条边权重为负，若它会成环也不能加入；权重最小化必须遵守生成树约束。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.046J Lecture 3: Minimum Spanning Trees I](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2012/6baf48ebe3babed294c31a907a916e08_MIT6_046JS12_lec03.pdf), pp. 5–7: cut exchange, Kruskal’s sorted scan, collective MST invariant and the lighter-crossing-edge contradiction.
- [Lecture 4: Minimum Spanning Trees II](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2012/bb54568c86c60d86b25e9c1111470384_MIT6_046JS12_lec04.pdf), pp. 1–2 and 4: disjoint-set operations and the precise safe-edge condition.
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_SpaingTrees.pdf|MIT 6.042J Spanning Trees slides]], PDF 3–6: component-colored cuts and the course’s Kruskal/Prim comparison. The finite multigraph, loop and componentwise extensions are justified explicitly above.
<!-- bilingual-en:end -->

- MIT 6.046J Lecture 3，第 5–7 页：割交换、Kruskal 排序扫描、全部已选边共同包含于一棵 MST 的不变量，以及更轻跨割边的反证。
- Lecture 4，第 1–2、4 页：并查集操作与安全边精确条件。
- MIT 6.042J Spanning Trees slides，PDF 3–6：分量着色所给的割，以及课程中的 Kruskal/Prim 对比。有限多重图、自环和逐分量扩展的依据已在上文说明。

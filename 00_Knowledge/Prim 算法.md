---
aliases:
  - Prim algorithm for minimum spanning trees
  - Prim 最小生成树算法
student_os: knowledge-atom
atom_id: MCS-GRAPH-032
atom_set: acyclic-graphs-trees-dags
atom_type: algorithm
status: source-checked
requires:
  - "[[MST 割性质]]"
related:
  - "[[Kruskal 算法]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

<!-- bilingual-en:start -->
# Prim’s algorithm
<!-- bilingual-en:end -->

*Prim 算法*

<!-- bilingual-en:start -->
Prim’s algorithm connects a new vertex to one growing tree at each step. It starts anywhere, looks at all edges leaving the current tree, and chooses a cheapest one. “Leaving” is crucial: a cheap edge wholly inside the tree cannot reach a new vertex, while a cheap edge between two outside vertices cannot attach to the tree yet. The aim is a [[最小生成树的定义与唯一性|minimum spanning tree (MST)]], minimizing the total weight of all selected edges.
<!-- bilingual-en:end -->

Prim 算法每步把一个新顶点接到正在生长的一棵树上。它从任意顶点开始，查看所有从当前树通向外部的边，并选其中最便宜的一条。“通向外部”是关键：便宜边若完全位于树内，就不能到达新顶点；若两端都在外面，也还不能把它接到当前树上。目标是[[最小生成树的定义与唯一性|最小生成树（MST）]]，即让全部所选边的权重和最小。

<!-- bilingual-en:start -->
Assume a finite, nonempty, connected [[图类型的建模边界|undirected graph]] $G=(V,E)$ with finite real edge weights $w(e)$. Zero, negative and tied weights are allowed. Keep parallel edges distinguishable; a self-loop never leaves a vertex set and is never eligible. Write $S$ for the vertices already in the growing tree and $F$ for its selected edges. They are different objects.
<!-- bilingual-en:end -->

设 $G=(V,E)$ 是有限、非空、连通的[[图类型的建模边界|无向图]]，每条边有有限实数权重 $w(e)$。允许零权、负权及并列，平行边按身份区分；自环永远不会跨出一个顶点集，所以没有资格被选择。用 $S$ 表示当前树中的顶点，用 $F$ 表示所选边，二者不是同一种对象。

<!-- bilingual-en:start -->
## Grow across the current boundary
<!-- bilingual-en:end -->

*跨越当前边界生长*

<!-- bilingual-en:start -->
Choose any root $r$, and initialize $S=\{r\}$ and $F=\varnothing$. As long as $S\ne V$, consider every edge with exactly one endpoint in $S$. Such edges cross the [[图割|cut]] $(S,V\setminus S)$, where $V\setminus S$ denotes the outside vertices. Choose one with minimum weight, say $e=uv$ with $u\in S$ and $v\notin S$, and add both $e$ to $F$ and $v$ to $S$. Ties may be broken arbitrarily.
<!-- bilingual-en:end -->

任选根 $r$，初始化 $S=\{r\}$、$F=\varnothing$。只要 $S\ne V$，就查看所有恰有一个端点在 $S$ 中的边。这些边跨越[[图割|割]] $(S,V\setminus S)$，其中 $V\setminus S$ 表示外部顶点。选一条权重最小的边 $e=uv$，$u\in S$、$v\notin S$，将 $e$ 加入 $F$，同时将 $v$ 加入 $S$。并列时可任选。

> [!example] Selected weights need not increase · 选中的边权不必递增
> <!-- bilingual-en:start -->
> Take four vertices $a,b,c,d$ and exactly these six undirected edges, with $ab$ denoting the edge joining $a,b$: $w(ab)=1$, $w(bc)=2$, $w(ac)=3$, $w(cd)=4$, $w(bd)=5$, $w(ad)=8$. Start at $d$. The first candidates are $cd,bd,ad$, so choose $cd$ of weight $4$. Now $S=\{c,d\}$; the crossing edges have weights $2,3,5,8$, so choose $bc$ of weight $2$. With $S=\{b,c,d\}$, choose $ab$ of weight $1$. The final tree has total $7$. The choices were $4,2,1$: Prim minimizes across the changing cut, not across a fixed global edge list.
> <!-- bilingual-en:end -->
>
> 取四个顶点 $a,b,c,d$，仅有以下六条无向边，其中 $ab$ 表示连接 $a,b$ 的边：$w(ab)=1$、$w(bc)=2$、$w(ac)=3$、$w(cd)=4$、$w(bd)=5$、$w(ad)=8$，从 $d$ 开始。第一步候选是 $cd,bd,ad$，所以选权重 $4$ 的 $cd$。此时 $S=\{c,d\}$，跨割边权为 $2,3,5,8$，故选权重 $2$ 的 $bc$。随后 $S=\{b,c,d\}$，选权重 $1$ 的 $ab$，总权重为 $7$。选择顺序是 $4,2,1$：Prim 每次在变化的割上取最小值，而不是按固定全局边表扫描。

<!-- bilingual-en:start -->
The next edge may leave any vertex of the current tree, not only the vertex most recently added. An unselected edge with both endpoints in $S$ would close a cycle with the existing tree path; an edge already selected is simply not a new candidate. This is why choosing the globally cheapest edge without checking its endpoints does not implement Prim.
<!-- bilingual-en:end -->

下一条边可以从当前树的任意顶点出发，不必从刚加入的顶点出发。一条尚未选入、但两端都在 $S$ 内的边，会与原有树路径闭环；已经选过的边则根本不是新候选。因此，不看端点位置就选全图最轻边，并不是 Prim 算法。

<!-- bilingual-en:start -->
## Why every step can lead to an MST
<!-- bilingual-en:end -->

*为什么每一步都能继续扩展成 MST*

<!-- bilingual-en:start -->
Two statements hold together: $(S,F)$ is a tree, and $F$ as a whole is contained in some MST of the input. They hold initially for the singleton and empty edge set. Adding one outside vertex by one edge keeps the selected graph connected and cannot create a cycle, because that vertex had no selected edge before.
<!-- bilingual-en:end -->

同时维持两项事实：$(S,F)$ 是树；整个 $F$ 包含于输入图的某棵 MST。初始单点树与空边集满足两者。每次用一条边加入一个外部顶点，保持连通；新顶点此前没有任何所选边，因此不可能新成环。

<!-- bilingual-en:start -->
All previous selected edges have both endpoints in $S$, so none crosses the current cut. The chosen edge is a minimum-weight crossing edge. The [[MST 割性质|safe-edge theorem]] therefore preserves the second statement as well. Its hypotheses include both the common MST containing $F$ and the cut crossing none of $F$; separate claims that each edge individually belongs to some MST would not suffice.
<!-- bilingual-en:end -->

此前选中的边两端都在 $S$ 中，没有边跨当前割；新选边又是跨割最轻边，所以[[MST 割性质|安全边定理]]也保持第二项事实。这里必须同时有“同一棵 MST 包含整个 $F$”和“割不跨 $F$ 的边”两个前提；每条边分别属于某棵 MST，并不足够。

<!-- bilingual-en:start -->
Connectivity guarantees a crossing edge whenever $S$ is a nonempty proper subset: follow an original path from a vertex in $S$ to a vertex outside, and take the first edge leaving $S$. Every step increases $|S|$ by one. Thus, for $n=|V|$, exactly $n-1$ additions reach all vertices. The final tree has $n-1$ edges and is contained in an MST with the same number of edges, so it equals that MST. For $n=1$, the initial zero-edge tree is already the answer.
<!-- bilingual-en:end -->

当 $S$ 非空但还不含全部顶点时，连通性保证存在跨割边：沿原图一条从 $S$ 内通往外部的路径，取第一次离开 $S$ 的那条边即可。每步都让 $|S|$ 增加一，因此 $n=|V|$ 时，恰好加入 $n-1$ 条边就覆盖全部顶点。最终树包含于某棵 MST，且与它都有 $n-1$ 条边，只能相同。$n=1$ 时，初始无边单点树已经是答案。

<!-- bilingual-en:start -->
## What a key measures, and when to restart
<!-- bilingual-en:end -->

*键值表示什么，何时需要重启*

<!-- bilingual-en:start -->
An implementation can keep, for each outside vertex $v$, the cheapest currently known edge from $S$ to $v$, together with that edge’s identity. If there is no such edge, use $+\infty$ as a bookkeeping marker. Select a vertex with the smallest finite key; after adding it, inspect its edges to outside vertices and improve their keys when the single edge weight is smaller. A min-priority queue is a data structure for repeatedly retrieving that smallest key. **The key is one attachment-edge weight, not the sum of weights along a path from the root.** This distinction separates Prim from shortest-path calculations.
<!-- bilingual-en:end -->

实现时，可以对每个外部顶点 $v$，记录当前从 $S$ 接到它的最便宜边及该边身份；若尚无这种边，用 $+\infty$ 作记账标记。取有限键值中最小的顶点，加入后检查它通往其他外部顶点的边，若单条边权更小就更新相应键值。最小优先队列是反复取出最小键值的数据结构。**这里的键值是一条接入边的权重，不是从根出发沿整条路径的权重和。**这正是 Prim 与最短路计算的区别。

> [!warning] An empty boundary need not mean the whole graph is covered · 边界为空不一定覆盖了全图
> <!-- bilingual-en:start -->
> On a disconnected input, one run stops when the root’s original connected component has been spanned. No crossing edge can reach a different component. To obtain a minimum spanning forest of the entire graph, restart from an uncovered vertex and repeat until every component, including isolated vertices, is covered. No MST of the disconnected whole graph exists. With no vertices there is no root to choose; a componentwise driver may return the empty forest.
> <!-- bilingual-en:end -->
>
> 输入不连通时，一次运行在覆盖根所在的原连通分量后停止，没有跨割边能够通向另一分量。若要得到全图的最小生成森林，就从尚未覆盖的顶点重新开始，直到包括孤立顶点在内的每个分量都覆盖；不连通全图本身不存在 MST。没有顶点时不能选根，逐分量运行的外层流程可返回空森林。

<!-- bilingual-en:start -->
Equal weights can change the tree selected by the root or tie-breaking, while the optimum weight remains the same; ties do not force multiple MSTs. Negative weights cause no difficulty because the proof uses comparisons, not nonnegative path lengths. [[Kruskal 算法|Kruskal]] instead maintains all forest components and merges them in global sorted order. Both methods use the same cut principle, applied to different current cuts.
<!-- bilingual-en:end -->

并列权重可能让不同起点或并列选择产生不同树，但最优总权重相同；出现并列不代表必有多个 MST。负权也没有问题，因为证明使用的是权重比较，不是路径长度非负性。[[Kruskal 算法]]则同时维护全部森林分量，按全局排序合并。两者使用同一割原理，只是当前割的来源不同。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.046J Lecture 4: Minimum Spanning Trees II](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2012/bb54568c86c60d86b25e9c1111470384_MIT6_046JS12_lec04.pdf), pp. 4–6: safe-edge hypotheses, the growing component, correctness, and the one-edge key update in the priority-queue implementation.
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_SpaingTrees.pdf|MIT 6.042J Spanning Trees slides]], PDF 3–6: gray-edge growth and the single-tree Prim viewpoint.
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=542|MIT Mathematics for Computer Science, PDF 542]], Algorithm 1: the current-tree boundary. The general tied-weight proof above uses the safe-edge theorem, rather than extending the textbook’s distinct-weight “every MST” wording.
<!-- bilingual-en:end -->

- MIT 6.046J Lecture 4，第 4–6 页：安全边前提、单分量增长、正确性及优先队列实现中的单边键值更新。
- MIT 6.042J Spanning Trees slides，PDF 3–6：灰边增长与 Prim 生长单棵树的视角。
- MIT《Mathematics for Computer Science》，PDF 542，算法 1：当前树的边界。正文允许并列的证明使用安全边定理，没有把教材在互异权重前提下的“每棵 MST”措辞直接推广。

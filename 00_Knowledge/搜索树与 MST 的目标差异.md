---
aliases:
  - Search trees versus minimum spanning trees
  - 搜索树与 MST 的区别
student_os: knowledge-atom
atom_id: MCS-GRAPH-029
atom_set: acyclic-graphs-trees-dags
atom_type: distinction
status: source-checked
requires:
  - "[[搜索标记与父森林]]"
  - "[[BFS 最短路树的顺序依赖]]"
  - "[[DFS 可达性与非最短路径]]"
  - "[[生成树的存在条件]]"
  - "[[最小生成树的定义与唯一性]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

<!-- bilingual-en:start -->
# Search trees and minimum spanning trees
<!-- bilingual-en:end -->

*搜索树与 MST 的目标差异*

<!-- bilingual-en:start -->
A tree can record how we reached each vertex, or it can specify which connections are cheapest to build for the whole network. These are different questions. A search tree organizes routes around a chosen source; a minimum spanning tree compares the total cost of all its edges. The word “tree” alone does not tell us which quantity was optimized.
<!-- bilingual-en:end -->

一棵树可以记录我们如何到达每个顶点，也可以指定整个网络怎样建设连线最便宜。这是两个不同问题：搜索树围绕选定源点组织路线，最小生成树则比较全部树边的总费用。单凭“树”这个词，不能判断优化了哪个量。

<!-- bilingual-en:start -->
For the comparison, use a finite, nonempty, connected [[图类型的建模边界|undirected graph]] $G=(V,E)$. A [[生成树|spanning tree]] keeps every vertex, is connected and has no cycle. Give edge $e$ a finite real weight $w(e)$. An [[最小生成树的定义与唯一性|MST]] minimizes $w(T)=\sum_{e\in T}w(e)$ over spanning trees $T$. The optimization problem specifies no root, although any vertex can later be chosen as a root for displaying the resulting tree.
<!-- bilingual-en:end -->

为了比较，取有限、非空、连通的[[图类型的建模边界|无向图]] $G=(V,E)$。[[生成树|生成树]]保留全部顶点，连通且无环。每条边 $e$ 配有限实数权重 $w(e)$，[[最小生成树的定义与唯一性|MST]]在全部生成树 $T$ 中最小化 $w(T)=\sum_{e\in T}w(e)$。这个优化问题没有指定根，但选好树之后，仍可任选一个顶点作为展示用的根。

<!-- bilingual-en:start -->
## What the parent tree promises
<!-- bilingual-en:end -->

*父树保证什么*

<!-- bilingual-en:start -->
Start a complete search from a source $s$ with all vertices initially undiscovered. The [[搜索标记与父森林|first-discovery parent edge]] records which existing edge first reached each new vertex. Parents were discovered earlier, so following parent records cannot cycle; it eventually reaches $s$. A completed single-source search visits all vertices reachable from $s$. In the connected undirected setting, these parent edges therefore form a spanning tree. On a disconnected graph they span only the source component.
<!-- bilingual-en:end -->

从源点 $s$ 开始完整搜索，初始全部顶点都未发现。[[搜索标记与父森林|首次发现的父边]]记录哪个已有边首次到达了每个新顶点。父顶点总比子顶点更早发现，所以沿父记录不可能成环，最终回到 $s$。完整单源搜索会访问从 $s$ 可达的全部顶点；在这里的连通无向图中，父边因此形成生成树。图不连通时，它只覆盖源点所在分量。

<!-- bilingual-en:start -->
[[DFS 可达性与非最短路径|Depth-first search (DFS)]] supplies a route, without promising that it is shortest. [[BFS 分层与无权最短路|Breadth-first search (BFS)]] visits vertices by edge-count distance: each parent path has the fewest edges among routes from $s$ to that vertex. Its [[BFS 最短路树的顺序依赖|chosen shortest-path tree]] can depend on adjacency order even when distances are fixed. If numeric weights are attached to the graph but BFS ignores them, its guarantee remains about edge counts.
<!-- bilingual-en:end -->

[[DFS 可达性与非最短路径|深度优先搜索（DFS）]]给出路线，但不承诺最短。[[BFS 分层与无权最短路|广度优先搜索（BFS）]]按边数距离逐层访问，所以每条父路径都用最少的边从 $s$ 到达对应顶点。距离固定时，[[BFS 最短路树的顺序依赖|所选最短路树]]仍可能依赖邻接顺序。即使图上附有数值权重，只要 BFS 不读取这些权重，保证仍然只是边数最少。

> [!example] One-hop paths can make an expensive tree · 一步可达却可能整树很贵
> <!-- bilingual-en:start -->
> In a triangle on $s,a,b$, let $w(sa)=100$, $w(sb)=1$, $w(ab)=1$. BFS from $s$ first discovers both other vertices through the direct edges $sa,sb$, independently of which neighbor is listed first. Both parent paths have one edge, but the tree weighs $101$. The MST selects $sb,ab$ and weighs $2$; its path from $s$ to $a$ has two edges. Thus the BFS tree is not an MST, and the MST is not an edge-count shortest-path tree rooted at $s$.
> <!-- bilingual-en:end -->
>
> 在顶点为 $s,a,b$ 的三角形中，设 $w(sa)=100$、$w(sb)=1$、$w(ab)=1$。BFS 从 $s$ 出发，通过直达边 $sa,sb$ 首次发现另两个顶点，不受先列哪个邻居影响。两条父路径都只用一步，但树的总权重为 $101$。MST 选择 $sb,ab$，总权重只有 $2$；其中从 $s$ 到 $a$ 却要走两条边。因此 BFS 树不是 MST，而该 MST 也不是以 $s$ 为根的边数最短路树。

<!-- bilingual-en:start -->
## Weighted shortest paths still have a different objective
<!-- bilingual-en:end -->

*加权最短路仍是另一个目标*

<!-- bilingual-en:start -->
For this paragraph assume nonnegative weights, so ordinary weighted shortest-path distances are finite on the connected graph. A path’s weight is the sum of its edge weights. A **weighted shortest-path tree rooted at $s$** contains a minimum-weight path from $s$ to every vertex. This optimizes each root-to-vertex path, whereas the MST sum counts each selected tree edge once. Edges shared by several root paths are not counted several times in the MST objective.
<!-- bilingual-en:end -->

本段再假设权重非负，使连通图中的通常加权最短路距离都是有限的。路径权重是沿途边权之和。**以 $s$ 为根的加权最短路树**为每个顶点保留一条从 $s$ 出发的最小权重路径。它优化每条根到顶点的路径；MST 的总和则把每条所选树边只计一次，多条根路径共用的边不会在 MST 目标中重复计费。

> [!example] Shortest paths versus the cheapest network · 最短路线与最便宜网络
> <!-- bilingual-en:start -->
> Now let $w(sa)=2$, $w(sb)=2$, $w(ab)=1$. The shortest weighted routes from $s$ to $a$ and $b$ are the direct edges, each of weight $2$; going through the other vertex costs $3$. Therefore the shortest-path tree uses $sa,sb$ and weighs $4$. An MST instead uses $ab$ and one direct edge, weighing $3$. If it uses $sa,ab$, its route from $s$ to $b$ costs $3>2$; the other MST fails shortestness for $a$. This single example disproves both general implications between the two objectives.
> <!-- bilingual-en:end -->
>
> 现在改为 $w(sa)=2$、$w(sb)=2$、$w(ab)=1$。从 $s$ 到 $a,b$ 的加权最短路线都是权重 $2$ 的直达边，绕经另一个顶点要花 $3$。因此最短路树选 $sa,sb$，总权重 $4$。MST 则选 $ab$ 加任一条直达边，总权重 $3$。若选 $sa,ab$，其中从 $s$ 到 $b$ 的路径权重为 $3>2$；另一棵 MST 则不能给出到 $a$ 的最短路。同一个例子否定了两项目标之间两个方向的一般蕴含。

<!-- bilingual-en:start -->
## When the outputs happen to agree
<!-- bilingual-en:end -->

*什么时候输出恰好兼容*

<!-- bilingual-en:start -->
If all edges have the same weight $c$, every spanning tree has $n-1$ edges and hence total weight $(n-1)c$. Thus every spanning BFS or DFS tree is an MST, even if $c=0$ or $c<0$: this statement concerns the spanning-tree cost only. It does not claim that negative-weight shortest-walk problems are well-defined. Nor does it make every MST a BFS tree: in an all-unit triangle, a two-edge path is an MST but may take two steps from its endpoint to a vertex directly adjacent to that endpoint in the input.
<!-- bilingual-en:end -->

若全部边的权重同为 $c$，每棵生成树都有 $n-1$ 条边，总权重都是 $(n-1)c$。所以任何覆盖全图的 BFS 或 DFS 树都是 MST，连 $c=0$、$c<0$ 也成立；这只是关于生成树费用的结论，并没有声称负权最短游走问题有良好定义。它也不表示每棵 MST 都是 BFS 树：在三条边均为单位权的三角形中，两边组成的路径是 MST，但从路径端点到原图中直接相邻的某点，可能要走两步。

<!-- bilingual-en:start -->
The singleton case gives the same zero-edge tree for all these constructions. On disconnected undirected graphs, compare trees component by component: a one-source search cannot span other components, while a minimum spanning forest includes an MST for every original component. For directed graphs, search still follows arrows, but the ordinary undirected MST problem used here is not a directed-tree optimization problem. Finally, [[Kruskal 算法|Kruskal]] and [[Prim 算法|Prim]] choose edges to preserve minimum total weight; a root used by Prim is a construction choice, not a shortest-path requirement.
<!-- bilingual-en:end -->

单点图中，这些构造都给出同一棵无边树。不连通无向图则要逐分量比较：一次源点搜索不能覆盖其他分量，最小生成森林要为每个原连通分量提供一棵 MST。有向图中的搜索仍沿箭头进行，但本卡通常的无向 MST 问题不能直接当作有向树优化。最后，[[Kruskal 算法]]与[[Prim 算法]]的选边规则维护的是总权重最小；Prim 所用的根只是构造起点，不是最短路要求。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.006 Lecture 9: Breadth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/196a95604877d326c6586e60477b59d4_MIT6_006S20_lec9.pdf), pp. 3–4: fixed-source edge-count distance and parent-tree output.
- [Lecture 10: Depth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf), pp. 2–3: reachability, non-shortest parent routes and component coverage.
- [Lecture 11: Weighted Shortest Paths](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/aa57a9785adf925bc85c1920f53755a0_MIT6_006S20_lec11.pdf), pp. 1–3: path-weight sums and the fixed-source objective, distinguished from edge-count distance.
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=539|MIT Mathematics for Computer Science, §12.11.4, PDF 539]]: minimum total spanning-tree weight. The two triangles and equal-weight case are derived explicitly above.
<!-- bilingual-en:end -->

- MIT 6.006 Lecture 9，第 3–4 页：固定源点的边数距离与父树输出。
- Lecture 10，第 2–3 页：可达性、可能非最短的父路径与分量覆盖范围。
- Lecture 11，第 1–3 页：路径权重之和、固定源点目标及其与边数距离的区别。
- MIT《Mathematics for Computer Science》，§12.11.4，PDF 539：生成树全部边权之和的最小化；正文明确计算两个三角形和等权情形。

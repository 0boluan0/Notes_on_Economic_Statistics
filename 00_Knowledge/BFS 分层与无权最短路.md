---
aliases:
  - Breadth-first search and unweighted shortest paths
  - BFS distance layers
student_os: knowledge-atom
atom_id: MCS-GRAPH-014
atom_type: theorem
status: source-checked
part_of:
  - "[[图的基本结构、路径与遍历.canvas]]"
requires:
  - "[[搜索标记与父森林]]"
  - "[[图距离与可达性]]"
implies:
  - "[[BFS 最短路树的顺序依赖]]"
---

<!-- bilingual-en:start -->
# BFS layers and shortest paths
<!-- bilingual-en:end -->

*BFS 分层与无权最短路*

<!-- bilingual-en:start -->
Breadth-first search, abbreviated **BFS**, answers a simple question: starting at a vertex $s$, what is the fewest number of edges needed to reach each other vertex? It first finds vertices one edge away, then two edges away, and so on. Here “shortest” refers to [[图距离与可达性|edge-count distance]], rather than a separate travel time or cost attached to each edge.
<!-- bilingual-en:end -->

**广度优先搜索**，简称 **BFS**，回答一个直接的问题：从顶点 $s$ 出发，到每个其他顶点最少需要经过多少条边？它先找到一条边就能到达的点，再找到两条边才能到达的点，依此类推。这里的“最短”指[[图距离与可达性|按边数计算的距离]]，而非另外赋给边的时间或费用。

<!-- bilingual-en:start -->
Work with a finite graph and a chosen source $s$ in its vertex set $V$. In a directed graph, follow only outgoing edges; in an undirected graph, each edge can be followed in either direction. The graph need not be connected. Write $d[v]$ for the distance label stored by the algorithm, and $\operatorname{dist}(s,v)$ for the actual minimum number of edges. Initially set $d[s]=0$ and all other labels to $\infty$, meaning “not yet discovered.”
<!-- bilingual-en:end -->

考虑一个有限图，并在顶点集 $V$ 中选定源点 $s$。有向图只沿出边前进，无向图的边可以朝两个方向走；图不必连通。用 $d[v]$ 表示算法保存的距离标签，用 $\operatorname{dist}(s,v)$ 表示真正的最少边数。开始时令 $d[s]=0$，其他标签都是 $\infty$，暂时表示“尚未发现”。

<!-- bilingual-en:start -->
## How the queue creates layers
<!-- bilingual-en:end -->

*队列怎样产生层次*

<!-- bilingual-en:start -->
A **first-in, first-out queue** (FIFO queue) processes the vertex that has waited longest. Put $s$ in the queue. Repeatedly remove its front vertex $u$ and inspect every outgoing neighbor $v$. If $v$ is still undiscovered, set $d[v]=d[u]+1$, mark it discovered immediately, and append it to the back of the queue. An already discovered vertex keeps its label. This is the [[搜索标记与父森林|discovery marking]] rule: mark on insertion, rather than waiting until removal, so each vertex enters the queue only once.
<!-- bilingual-en:end -->

**先进先出队列**（FIFO queue）总是先处理等待最久的顶点。先把 $s$ 放入队列，然后反复取出队首 $u$，查看它的所有出邻居 $v$。若 $v$ 尚未发现，就令 $d[v]=d[u]+1$，立即标为已发现，并放到队尾。已经发现的顶点保留原标签。这就是[[搜索标记与父森林|发现时标记]]的规则：入队时便标记，而非等到出队，因而每个顶点只入队一次。

> [!example] A small layered graph · 一个分层小图
> <!-- bilingual-en:start -->
> Take directed edges $s\to a$, $s\to b$, $a\to t$, $b\to t$, and $t\to w$, together with an isolated vertex $z$. After processing $s$, the queue contains $a,b$ in some order, both labelled $1$. Processing either one first discovers $t$ with label $2$; the other does not overwrite that label. Processing $t$ discovers $w$ with label $3$. Thus the layers are $\{s\}$, $\{a,b\}$, $\{t\}$, $\{w\}$, while $z$ remains undiscovered. The same layers result if these five edges are undirected.
> <!-- bilingual-en:end -->
> 取有向边 $s\to a$、$s\to b$、$a\to t$、$b\to t$、$t\to w$，再加一个孤立顶点 $z$。处理 $s$ 后，队列以某种顺序放着 $a,b$，标签都是 $1$。先处理其中任何一个，都会以标签 $2$ 发现 $t$；另一个不会改写这个标签。处理 $t$ 后，又以标签 $3$ 发现 $w$。各层因此为 $\{s\}$、$\{a,b\}$、$\{t\}$、$\{w\}$，而 $z$ 始终未发现。把这五条边改为无向边，层次也相同。

<!-- bilingual-en:start -->
## Why first discovery is already shortest
<!-- bilingual-en:end -->

## 分层不变量

<!-- bilingual-en:start -->
The key invariant is that queue labels are nondecreasing from front to back. While vertices labelled $k$ are being processed, newly discovered vertices receive label $k+1$ and go behind all remaining label-$k$ vertices. Consequently, BFS finishes a whole layer before expanding the next one. This ordering alone must still be connected to the true distances; the following induction does that.
<!-- bilingual-en:end -->

关键不变量是：从队首到队尾，标签不会递减。处理标签为 $k$ 的顶点时，新发现的点得到标签 $k+1$，排在所有尚未处理的标签 $k$ 顶点之后。因此，BFS 总是完成一层，才开始扩展下一层。还需要证明这些标签确实等于真实距离，下面用归纳完成这一步。

<!-- bilingual-en:start -->
The base layer contains only $s$, whose distance is $0$. Assume all vertices at true distance at most $k$ have been correctly discovered, and layers below $k$ have been fully processed. When an edge from a layer-$k$ vertex $u$ first discovers $v$, it gives a route of length $k+1$. If $v$ had a shorter route, then either $v=s$ and was already marked, or a predecessor on that shorter route would have distance at most $k-1$. That predecessor would already have been processed and would have discovered $v$. Both alternatives contradict first discovery now. Thus $v$ really has distance $k+1$.
<!-- bilingual-en:end -->

基础层只有 $s$，它的距离为 $0$。假设真实距离不超过 $k$ 的顶点都已被正确发现，且小于 $k$ 的各层已经处理完。若从第 $k$ 层顶点 $u$ 的一条边首次发现 $v$，便得到一条长为 $k+1$ 的路线。若 $v$ 还有更短路线，那么要么 $v=s$、早已标记；要么该路线中 $v$ 的前驱距离至多为 $k-1$，早已被处理并应当发现了 $v$。这两种情况都与“此刻首次发现”矛盾，所以 $v$ 的真实距离就是 $k+1$。

<!-- bilingual-en:start -->
Conversely, every vertex at distance $k+1$ has a predecessor at distance $k$ on a shortest path: a shorter route to that predecessor would shorten the whole path. Processing every vertex in layer $k$ therefore discovers every vertex in layer $k+1$. Together with FIFO ordering, this proves the induction and the identity
<!-- bilingual-en:end -->

反过来，真实距离为 $k+1$ 的每个顶点，在一条最短路径上都有距离为 $k$ 的前驱：若到这个前驱还能更短，就能缩短整条路径。因此处理完第 $k$ 层，也就发现了第 $k+1$ 层的所有顶点。结合 FIFO 次序，归纳成立，并得到

$$L_k=\{v\in V:d[v]=k\}=\{v\in V:\operatorname{dist}(s,v)=k\},\qquad k=0,1,2,\ldots.$$

<!-- bilingual-en:start -->
Because the graph is finite and each vertex is queued at most once, the search terminates. Every discovered vertex has an actual route from $s$, and the induction finds every reachable vertex. Therefore **after the queue is exhausted**, labels left at $\infty$ mean unreachable. Before completion, an infinite label only means not discovered yet. One complete BFS gives distances to all vertices from one source. Self-loops and parallel edges do not change this reasoning: they may cause repeated inspections, but not repeated discoveries. Recording one discovering edge also gives a [[BFS 最短路树的顺序依赖|shortest-path parent tree]].
<!-- bilingual-en:end -->

图是有限的，每个顶点至多入队一次，所以搜索会终止。每个已发现的点都有从 $s$ 出发的实际路线，归纳又保证所有可达点都会被找到。因此，**队列处理完之后**，仍为 $\infty$ 的标签才表示确实不可达；搜索尚未完成时，它只表示尚未发现。一次完整 BFS 就给出从一个源点到所有顶点的距离。自环和平行边不改变这个论证：可能重复查看邻接项，但不会重复发现顶点。若同时记录首次发现所用的边，还会得到一棵[[BFS 最短路树的顺序依赖|最短路父树]]。

<!-- bilingual-en:start -->
## When edge weights change the question
<!-- bilingual-en:end -->

## 权重边界

<!-- bilingual-en:start -->
BFS minimizes the number of edges. If every edge has the same **positive** weight $c$, a route with $k$ edges has cost $ck$, so minimizing cost is equivalent to minimizing $k$. Positivity is essential for this equivalence. If $c=0$, every existing route costs zero: the BFS route is a minimum-cost route, but longer routes tie with it. If $c<0$, more edges can lower the cost. For instance, in the directed triangle $s\to t$, $s\to a\to t$, with every edge weighted $-1$, BFS chooses the one-edge route of cost $-1$ although the two-edge route costs $-2$.
<!-- bilingual-en:end -->

BFS 最小化的是边数。若每条边的权重都是同一个**正数** $c$，那么含 $k$ 条边的路线成本为 $ck$，最小化成本与最小化 $k$ 等价。这个等价需要正性。若 $c=0$，每条存在的路线成本都是零：BFS 找到的路线确实成本最小，但更长路线也并列最优。若 $c<0$，更多边反而可能降低成本。例如有向三角形 $s\to t$、$s\to a\to t$ 的每条边权都是 $-1$，BFS 选择成本 $-1$ 的一条边路线，但两条边路线的成本为 $-2$。

<!-- bilingual-en:start -->
With unequal positive weights, a direct edge of weight $100$ is discovered before a three-edge route whose edges each weigh $1$, even though $100>3$. Thus general weighted shortest paths require a method suited to the weight assumptions, such as DAG shortest paths, Dijkstra's algorithm, or Bellman–Ford. The [[图类型的建模边界|graph and weight model]] determines which distance question is being asked.
<!-- bilingual-en:end -->

即使权重全为正，只要它们不同，也会出问题：一条权重 $100$ 的直达边会先于三条权重各为 $1$ 的路线发现终点，但 $100>3$。因此一般加权最短路要按权重条件选用适合的方法，例如 DAG 最短路、Dijkstra 或 Bellman–Ford。所采用的[[图类型的建模边界|图与权重模型]]决定了正在求哪一种距离。

> [!question]- Why does FIFO matter? · FIFO 为什么重要？
> <!-- bilingual-en:start -->
> Replacing the queue by a last-in, first-out stack can let a deep branch discover a vertex before a shorter branch is explored. First discovery then has no shortest-distance guarantee, as illustrated by [[DFS 可达性与非最短路径|depth-first search]]. FIFO supplies the ordering used in the proof; it is not merely a storage preference.
> <!-- bilingual-en:end -->
> 若把队列换成后进先出的栈，较深分支可能在较短分支被探索前就发现某个顶点，此时首次发现不再保证最短距离，[[DFS 可达性与非最短路径|深度优先搜索]]便说明了这一点。FIFO 提供的是证明所需的处理顺序，不只是存储偏好。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.006, Lecture 9: Breadth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/196a95604877d326c6586e60477b59d4_MIT6_006S20_lec9.pdf), pp. 1–4: directed/undirected adjacency, edge-count distance, layer induction and parent representation.
- [MIT 6.006, Recitation 13: Breadth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/6d8aaabf67d391877f978172b293d7be_MIT6_006F11_rec13.pdf), pp. 1–2: reachability conventions, the FIFO queue and its equivalence to processing successive levels.
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=437|MIT Mathematics for Computer Science, §10.3.1]], together with PDF pp. 434–435: edge-count shortest distance, removing repeated vertices from a shortest walk, and the finite $|V|-1$ bound for reachable distinct vertices. The equal-weight cases above follow by comparing the costs $ck$ directly.
<!-- bilingual-en:end -->

- MIT 6.006 Lecture 9，第 1–4 页：有向/无向邻接、边数距离、逐层归纳与父指针表示。
- MIT 6.006 Recitation 13，第 1–2 页：可达性约定、FIFO 队列与按层处理的等价关系。
- MIT Mathematics for Computer Science §10.3.1，PDF 第 437 页及第 434–435 页：边数最短距离、最短游走去重复以及有限图中可达不同顶点的 $|V|-1$ 上界；上面的等权边界由成本 $ck$ 直接比较得到。

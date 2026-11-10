---
aliases:
- Kahn algorithm and cycle certificate
- Kahn 拓扑排序
student_os: knowledge-atom
atom_id: MCS-GRAPH-034
atom_set: acyclic-graphs-trees-dags
atom_type: algorithm
status: source-checked
requires:
- '[[DAG 与拓扑序等价]]'
- '[[度数、入度与出度]]'
related:
- '[[有向 DFS 的环证书]]'
- '[[拓扑序的线性扩张]]'
part_of:
- '[[无环图：树、生成树、DAG 与拓扑排序.canvas]]'
---

<!-- bilingual-en:start -->
# Kahn's topological sort
<!-- bilingual-en:end -->

*Kahn 拓扑排序与环证书*

<!-- bilingual-en:start -->
A task can be listed next when none of its unfinished prerequisites remain. **Kahn's algorithm** repeatedly chooses such a task, records it, and removes the constraints it supplied to later tasks. Either this lists every vertex in a valid order, or it stops with a nonempty remainder that contains a directed cycle.
<!-- bilingual-en:end -->

一项任务的尚未完成先修条件都消失后，它就可以排在下一位。**Kahn 算法**反复选取这样的任务，记录它，再移除它对后续任务施加的约束。结果要么把全部顶点排成合法顺序，要么停在一个含有向环的非空剩余部分。

<!-- bilingual-en:start -->
Use a finite [[图类型的建模边界|directed graph]] $G=(V,E)$, where $V$ contains the vertices and $E$ the arrows. A **[[拓扑序|topological order]]** lists every vertex exactly once and puts $u$ before $v$ whenever $u\to v$ is an edge. Such an order exists exactly when the graph is a [[有向无环图|DAG]], meaning it has no directed cycle; the existence proof is in [[DAG 与拓扑序等价|DAGs and topological orders]]. Here we explain the actual construction and what a failed construction certifies.
<!-- bilingual-en:end -->

取有限[[图类型的建模边界|有向图]] $G=(V,E)$，其中 $V$ 是顶点集，$E$ 是箭头集合。**[[拓扑序]]**把每个顶点恰好列出一次，并要求每条边 $u\to v$ 的起点 $u$ 排在终点 $v$ 前。它存在当且仅当图是 [[有向无环图|DAG]]，也就是没有有向环；存在性证明见[[DAG 与拓扑序等价]]。本卡解释具体构造，以及构造失败究竟证明什么。

<!-- bilingual-en:start -->
## Count only the prerequisites still present
<!-- bilingual-en:end -->

*只统计尚未移除的先修边*

<!-- bilingual-en:start -->
At any stage, the remaining graph keeps the unlisted vertices and all original arrows between them. Let $d(v)$ be the [[度数、入度与出度|in-degree]] of an unlisted vertex $v$ in this remaining graph: the number of edges entering it from other remaining vertices, with a self-loop also counted. A vertex with $d(v)=0$ is a **source**, or ready vertex. The count changes as vertices are removed; it is not permanently equal to the original in-degree.
<!-- bilingual-en:end -->

每个阶段的剩余图保留尚未输出的顶点，以及它们之间的全部原有箭头。令 $d(v)$ 为尚未输出顶点 $v$ 在剩余图中的[[度数、入度与出度|入度]]：数仍然存在的进入它的边，自环也计入。$d(v)=0$ 的顶点称为**源点**，也就是已就绪顶点。顶点被移除时这个计数会变化，不能一直沿用原图入度。

<!-- bilingual-en:start -->
Initially compute all in-degrees and put **every** zero-in-degree vertex into a ready collection. Repeat: take one ready vertex $u$, append it to the output, and for each edge $u\to v$ subtract one from $d(v)$. This decrement is the bookkeeping equivalent of deleting that edge. If the count becomes zero, add $v$ to the ready collection. Each vertex is output at most once. The input graph itself need not be physically edited.
<!-- bilingual-en:end -->

先计算所有初始入度，把**全部**零入度顶点放入就绪集合。之后反复取出一个就绪顶点 $u$，把它加入输出；对每条出边 $u\to v$，令 $d(v)$ 减一，这就是删除该边的计数记账。若计数变为零，就把 $v$ 加入就绪集合。每个顶点至多输出一次，不必真的修改输入图。

> [!example] Watch the remaining counts · 跟着剩余入度走一遍
> <!-- bilingual-en:start -->
> Take $a\to c$, $b\to c$, $b\to d$, and $c\to d$. Initially the counts in order $(a,b,c,d)$ are $(0,0,2,2)$, so $a,b$ are ready. Choose $a$ first: only $c$ changes, from 2 to 1. Choose $b$ next: $c$ changes from 1 to 0 and $d$ from 2 to 1. Now choose $c$, which makes $d$ ready; then choose $d$. The output is $a,b,c,d$. Choosing $b$ before $a$ instead gives $b,a,c,d$, also valid.
> <!-- bilingual-en:end -->
>
> 取边 $a\to c$、$b\to c$、$b\to d$、$c\to d$。按 $(a,b,c,d)$ 排列的初始入度为 $(0,0,2,2)$，因此 $a,b$ 已就绪。先选 $a$，只有 $c$ 从2减为1；再选 $b$，$c$ 从1减为0，$d$ 从2减为1。接着选 $c$，使 $d$ 就绪，再选 $d$，得到 $a,b,c,d$。若先选 $b$ 再选 $a$，则得到同样合法的 $b,a,c,d$。

<!-- bilingual-en:start -->
A first-in, first-out (FIFO) queue is one convenient ready collection, but FIFO order is not needed for correctness here. Any currently ready vertex is a valid next choice. This differs from the distance-layer role of FIFO in BFS. Different choices can give [[拓扑序的线性扩张|different legal topological orders]]; the algorithm is not promising a unique schedule.
<!-- bilingual-en:end -->

先进先出（FIFO）队列是就绪集合的一种方便实现，但这里的正确性不依赖先进先出；任何当前就绪顶点都可以成为下一项。这与 BFS 用 FIFO 维持距离层的作用不同。不同选择可能给出[[拓扑序的线性扩张|不同的合法拓扑序]]，算法没有承诺唯一的调度顺序。

<!-- bilingual-en:start -->
## Why a complete output is valid
<!-- bilingual-en:end -->

*为什么完整输出一定合法*

<!-- bilingual-en:start -->
The invariant is that $d(v)$ equals the number of incoming edges whose tails have not yet been output. It holds initially. Outputting $u$ removes exactly its outgoing edges, and the decrements maintain it. Thus a vertex can be output only after all of its incoming predecessors have been output. For every edge $u\to v$, $u$ therefore comes before $v$. If all vertices have been output, this is a topological order.
<!-- bilingual-en:end -->

不变量是：$d(v)$ 恰好等于起点尚未输出的入边数。初始计算使它成立；输出 $u$ 时，恰好删除它的全部出边，相应减一维持这个不变量。因此，一个顶点只有在全部入边的前驱都已输出后才能输出。对任意边 $u\to v$，$u$ 必在 $v$ 前。如果全部顶点都输出了，就得到了拓扑序。

<!-- bilingual-en:start -->
Conversely, a nonempty remaining subgraph of a DAG always has a source. Hence the ready collection cannot become empty before a DAG has been completely output. This links the implementation to the [[DAG 与拓扑序等价|nonempty induced-subgraph condition]] rather than assuming that the original graph merely had one source.
<!-- bilingual-en:end -->

反过来，DAG 的每个非空剩余子图都有源点，所以还未输出完一个 DAG 时，就绪集合不会变空。这把实现连接到[[DAG 与拓扑序等价|每个非空诱导子图都有源点]]的条件，而不只是要求原图曾经有一个源点。

<!-- bilingual-en:start -->
## What an early stop proves
<!-- bilingual-en:end -->

*提前停住究竟证明什么*

<!-- bilingual-en:start -->
Suppose the ready collection is empty but at least one vertex remains. Every remaining vertex then has an incoming edge from the remainder. Pick $w_0$ there and repeatedly choose a predecessor $w_{i+1}\to w_i$. Finiteness forces a repeated vertex. If the first repetition is $w_j=w_i$ with integers $0\le i<j$, the repeated segment gives a directed cycle. When $j=i+1$, the edge is a self-loop. Otherwise, read the segment in reverse:
<!-- bilingual-en:end -->

假设就绪集合已空，但仍有顶点未输出。此时每个剩余顶点都有来自剩余集合的入边。选其中一个 $w_0$，不断选择前驱 $w_{i+1}\to w_i$。有限性迫使重复发生；若第一次重复为 $w_j=w_i$，下标是整数且 $0\le i<j$，重复段就给出有向环。$j=i+1$ 时就是一条自环；否则反向读这段：

$$w_i=w_j\to w_{j-1}\to\cdots\to w_{i+1}\to w_i.$$

<!-- bilingual-en:start -->
The reverse reading matters: the predecessor search walked against the arrows. A residual subgraph with every in-degree positive is already a checkable certificate that the graph is cyclic. To return the actual cycle as a vertex/edge list, record the predecessor choices and extract the repeated segment; merely reporting “the ready collection is empty” has not yet listed that cycle.
<!-- bilingual-en:end -->

反向读很重要：选择前驱时，实际是逆着箭头追踪。剩余子图中每点入度都为正，已经是可核查的“图含环”证书；若接口要求返回具体的顶点或边列表，就还须记录前驱选择并提取重复段。仅报告“就绪集合为空”并没有列出那条环。

> [!warning] The remainder need not consist only of cycle vertices · 剩余点不一定都在环上
> <!-- bilingual-en:start -->
> Let $a\to b$, $b\to a$, and $b\to c$, together with an isolated vertex $d$. Only $d$ is initially ready. After outputting it, the remainder is $\{a,b,c\}$ and has no source. The cycle is $a\to b\to a$; $c$ remains because its prerequisite is blocked, not because $c$ belongs to a cycle. The output prefix $[d]$ is not a topological order of the whole graph.
> <!-- bilingual-en:end -->
>
> 取 $a\to b$、$b\to a$、$b\to c$，再加孤立顶点 $d$。起初只有 $d$ 就绪；输出它后，剩余集合为 $\{a,b,c\}$，没有源点。其中的环是 $a\to b\to a$；$c$ 留下是因为先修被阻塞，不是因为 $c$ 本身位于环上。输出前缀 $[d]$ 不能称为整张图的拓扑序。

<!-- bilingual-en:start -->
## Boundary cases and running time
<!-- bilingual-en:end -->

*边界与运行成本*

<!-- bilingual-en:start -->
An empty graph succeeds with an empty output. An edgeless graph lists all its vertices in any chosen order. A self-loop contributes an incoming edge that cannot disappear before its own vertex is removed, so that vertex cannot become ready. If parallel edges are allowed, count and decrement **each edge copy**; alternatively, consistently deduplicate the graph before both steps. Mixing neighbor counts with per-edge decrements is incorrect. Edge weights do not affect these precedence constraints.
<!-- bilingual-en:end -->

空图以空输出成功；无边图可以按任意选择顺序输出全部顶点。自环贡献一条入边，而在自身顶点移除之前它不会消失，因此该顶点无法就绪。若允许平行边，计数与递减都必须逐一处理**每份边**；也可以先一致地去重，再做两步。不能按邻居数初始化，却按边副本数递减。边权不改变这里的先后约束。

<!-- bilingual-en:start -->
With [[邻接表与邻接矩阵|outgoing adjacency lists]], constant-time access to vertex counts, and constant-time queue insertion/removal, initialization and all edge scans take $O(|V|+|E|)$ time for nonempty inputs, where $|V|$ and $|E|$ count vertices and edges. Each vertex enters the ready queue at most once, and each edge causes at most one decrement. Auxiliary counts, queue and output use $O(|V|)$ space beyond the input. If a concrete cycle is requested after failure, scan the original edges once more, retaining only edges with both ends remaining and saving one incoming predecessor for each remaining vertex. Following these saved predecessors until a repetition takes $O(|V|)$ further time. This keeps the same overall time and auxiliary-space bounds. An empty input still has constant setup cost. Repeatedly rescanning every vertex to find the next source, or using a priority queue to demand the smallest label, needs its own cost analysis.
<!-- bilingual-en:end -->

采用[[邻接表与邻接矩阵|出邻接表]]、常数时间访问顶点计数，以及常数时间入队和出队时，初始化与全部边扫描对非空输入总共耗时 $O(|V|+|E|)$，其中 $|V|,|E|$ 分别是顶点数与边数。每个顶点至多入队一次，每条边至多引起一次递减；输入之外的计数、队列和输出占 $O(|V|)$ 空间。若失败后还要求具体环，可再扫描原边一次，只考虑两端都剩余的边，并为每个剩余顶点保存一条入边的前驱。沿这些前驱追到重复，再花 $O(|V|)$ 时间，总时间与辅助空间界仍不变。空输入仍有常数级设置成本。若每次扫描全部顶点寻找源点，或使用优先队列强求最小标签，就需要另算成本。

> [!question]- What does a partial output tell you? · 部分输出告诉我们什么？
> <!-- bilingual-en:start -->
> If 8 vertices have been output but 3 remain and no source is ready, the list of 8 is not a full topological order. The remaining 3 vertices contain a directed cycle; they need not all lie on that cycle. Obtain a concrete cycle by the predecessor extraction above.
> <!-- bilingual-en:end -->
>
> 若已输出8个顶点，仍有3个顶点且没有就绪源点，这8个点的列表不是完整拓扑序。剩余3个点中存在有向环，但不一定每点都在环上；可用前面的前驱提取法给出具体环。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=442|MIT Mathematics for Computer Science (2018), §10.5.1, PDF442–443]]: repeatedly choose a minimal/source vertex in the remaining DAG to build a topological list.
- Robert Sedgewick and Kevin Wayne, [Algorithms, 4th edition, TopologicalX.java](https://algs4.cs.princeton.edu/42digraph/TopologicalX.java.html), the unweighted `TopologicalX(Digraph)` constructor and its specification: residual in-degree array, all-source initialization, queue updates, incomplete-output detection, and the linear time/space bounds. The concrete cycle-extraction argument is given above.
- [[01_Math/07-Mathematics for Computer Science/02_Structures.md#17.1 DAG 与拓扑排序|Permanent Session 17.1]]: source deletion and the finite no-source cycle argument.
<!-- bilingual-en:end -->

- MIT《Mathematics for Computer Science》（2018）§10.5.1，PDF442–443：在剩余 DAG 中反复选择 minimal/source 顶点，构造拓扑列表。
- Sedgewick 与 Wayne，《Algorithms》第4版配套 `TopologicalX.java`，未加权构造器及其说明：剩余入度数组、全部源点初始化、队列更新、输出不完整判定及线性时间/空间界。正文另展开具体环的提取证明。
- 永久 Session17.1：删除源点与有限无源图产生环的论证。

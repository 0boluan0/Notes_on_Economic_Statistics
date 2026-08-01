---
aliases:
  - "Acyclic Graphs: Trees, Spanning Trees, DAGs, and Topological Sort"
  - "Trees and Spanning Trees"
  - "Directed Acyclic Graphs"
  - "DAG"
  - "Topological Sort"
  - "最小生成树"
status: source-checked
---

# 无环图：树、生成树、DAG 与拓扑排序
<!-- bilingual-en:start -->
*Acyclic graphs: trees, spanning trees, DAGs, and topological sorting*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 没有 cycle 会把一般图压缩成可控制的骨架。无向 connected acyclic graph 是 tree，提供唯一 path 和最小连通骨架；有向 acyclic graph 是 DAG，提供不违背依赖的 topological order。
> **具体锚点：** 通信网络中，spanning tree 用 $n-1$ 条边连接全部站点；课程先修 DAG 中，反复选 indegree 为 0 的课程就得到合法修课顺序。
> **核心难点：** tree 与 DAG 共享“无环”却不是同一对象：tree 的边无方向且必须 connected，DAG 可以断开并允许两点间多条有向 path；$n-1$ 条边、topological order 和 MST 的结论各有附加条件。
> **为什么重要：** 唯一路径支持递归、广播和证明；spanning tree/MST 支持最低成本连通；DAG/topological sort 支持依赖解析、构建系统和并行调度。
> **继续：** 先看 [[#同一个“无环”，两个不同分支]]，无向分支进入 [[#Tree 的等价刻画]] 与 [[#生成树和最小生成树]]；有向分支进入 [[#DAG 与拓扑排序]]。
> <!-- bilingual-en:start -->
> **What it solves:** Removing cycles turns a general graph into a controlled skeleton. An undirected connected acyclic graph is a tree, with unique paths and a minimal connected backbone. A directed acyclic graph is a DAG and admits a topological order that respects dependencies.
> **Concrete anchor:** In a communication network, a spanning tree connects every station with $n-1$ edges. In a prerequisite DAG, repeatedly selecting a course of in-degree zero produces a valid completion order.
> **Central difficulty:** Trees and DAGs share acyclicity but are not the same object. Tree edges are undirected and the graph must be connected; a DAG may be disconnected and can contain several directed paths between two vertices. Claims involving $n-1$ edges, topological order, and MSTs each require additional conditions.
> **Why it matters:** Unique paths support recursion, broadcasting, and proofs; spanning trees and MSTs support low-cost connectivity; DAGs and topological sorting support dependency resolution, build systems, and parallel scheduling.
> **Continue with:** Start with the [[#同一个“无环”，两个不同分支|two branches of acyclicity]]. Follow the undirected branch through [[#Tree 的等价刻画|tree characterizations]] and [[#生成树和最小生成树|spanning trees and MSTs]], or the directed branch through [[#DAG 与拓扑排序|DAGs and topological sorting]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf]] Sections 10.5 and 12.11：核验 DAG/topological scheduling、tree 等价刻画、forest 边数、spanning tree、cut property 与 MST algorithms。
> - MIT OpenCourseWare 6.006, [Lecture 10: Depth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf)：核验 reverse finishing order、cycle detection 与 topological-sort 算法接口。
> - [[01_Math/07-Mathematics for Computer Science/02_Structures.md#Session 17 — Directed Acyclic Graphs]] 与 [[01_Math/07-Mathematics for Computer Science/02_Structures.md#Session 21 — Trees and Minimum Spanning Trees]]：核对课程符号、例题、边界与证明路径。
> <!-- bilingual-en:start -->
> - Sections 10.5 and 12.11 of [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] support DAG scheduling and topological order, equivalent tree characterizations, forest edge counts, spanning trees, the cut property, and MST algorithms.
> - MIT OpenCourseWare 6.006, [Lecture 10: Depth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf), supports reverse finishing order, cycle detection, and the algorithmic interface to topological sorting.
> - The local course sections on [[01_Math/07-Mathematics for Computer Science/02_Structures.md#Session 17 — Directed Acyclic Graphs|directed acyclic graphs]] and [[01_Math/07-Mathematics for Computer Science/02_Structures.md#Session 21 — Trees and Minimum Spanning Trees|trees and minimum spanning trees]] establish the course notation, examples, boundaries, and proof route.
> <!-- bilingual-en:end -->

## 同一个“无环”，两个不同分支
<!-- bilingual-en:start -->
*One notion of “no cycles,” two different branches*
<!-- bilingual-en:end -->

| 分支 | 定义 | 无环带来的核心结构 | 典型任务 |
|---|---|---|---|
| undirected tree/forest | 无向且 acyclic；tree 还要求 connected | 两点 unique path、删边断开、$n-c$ 条边 | 连通骨架、MST、递归分解 |
| directed DAG | 没有 directed cycle | source/sink 存在、topological order、依赖偏序 | 先修关系、构建顺序、调度 |
<!-- bilingual-en:start -->
| Branch | Definition | Structure created by acyclicity | Typical task |
|---|---|---|---|
| Undirected tree or forest | Undirected and acyclic; a tree is also connected | Unique paths, disconnection after deleting a tree edge, and $n-c$ edges | Connected backbones, MSTs, recursive decomposition |
| Directed DAG | Contains no directed cycle | Sources and sinks, topological order, and dependency order | Prerequisites, build order, scheduling |
<!-- bilingual-en:end -->

把一棵 undirected tree 的边任意定向会得到 DAG，但一般 DAG 的 underlying undirected graph 可以有 cycle。例如 $A\to B,A\to C,B\to D,C\to D$ 没有 directed cycle，却在忽略方向后形成四边环。因此 tree 定理不能只因看见“DAG”就套用。
<!-- bilingual-en:start -->
Orienting the edges of an undirected tree in any way produces a DAG, but the underlying undirected graph of a general DAG may contain a cycle. For example, $A\to B,A\to C,B\to D,C\to D$ has no directed cycle, yet ignoring orientation produces a four-cycle. Tree theorems therefore do not follow merely from seeing the label “DAG.”
<!-- bilingual-en:end -->

## Tree 的等价刻画
<!-- bilingual-en:start -->
*Equivalent characterizations of a tree*
<!-- bilingual-en:end -->

对 finite simple undirected graph $T$，下列条件等价：
<!-- bilingual-en:start -->
For a finite simple undirected graph $T$, the following conditions are equivalent:
<!-- bilingual-en:end -->

1. $T$ connected 且 acyclic；
2. 任意两顶点间恰有一条 simple path；
3. $T$ connected，且每条边都是 bridge；
4. $T$ 是 edge-minimal connected graph；
5. $T$ acyclic，且在任意两个不相邻顶点间加边都会产生 cycle；
6. 若 $|V|=n$，则 $T$ connected 且 $|E|=n-1$；等价地，$T$ acyclic 且 $|E|=n-1$。
<!-- bilingual-en:start -->

&nbsp;
**1.** $T$ is connected and acyclic.<br>
**2.** There is exactly one simple path between every pair of vertices.<br>
**3.** $T$ is connected and every edge is a bridge.<br>
**4.** $T$ is edge-minimal among connected graphs.<br>
**5.** $T$ is acyclic and adding an edge between any nonadjacent pair creates a cycle.<br>
**6.** If $|V|=n$, then $T$ is connected with $|E|=n-1$; equivalently, it is acyclic with $|E|=n-1$.<br>
<!-- bilingual-en:end -->

“unique path”把 connected 与 acyclic 联在一起：connected 保证至少一条 path；若有两条不同 simple paths，它们从首次分叉到再次汇合的两段组成 cycle。反过来，cycle 上任意两点沿两个方向给出两条不同 paths。
<!-- bilingual-en:start -->
The unique-path property unites connectivity and acyclicity. Connectivity guarantees at least one path. If two distinct simple paths exist, the two segments from their first divergence to their next reunion form a cycle. Conversely, two vertices on a cycle have distinct paths around its two sides.
<!-- bilingual-en:end -->

$n-1$ 条边不能单独证明 tree：一个 triangle 加一个 isolated vertex 有 4 个顶点、3 条边，却既不 connected 又含 cycle。边数条件必须与 connected 或 acyclic 至少一个结构条件合用。
<!-- bilingual-en:start -->
Having $n-1$ edges alone does not prove that a graph is a tree. A triangle plus an isolated vertex has four vertices and three edges, yet is disconnected and contains a cycle. The edge count must be combined with either connectivity or acyclicity.
<!-- bilingual-en:end -->

forest 是任意 acyclic undirected graph；每个 component 是 tree。若 forest 有 $n$ 个顶点、$c$ 个 components，则
<!-- bilingual-en:start -->
A forest is any acyclic undirected graph, and each component is a tree. If a forest has $n$ vertices and $c$ components, then
<!-- bilingual-en:end -->

$$
|E|=n-c.
$$

至少两个顶点的 finite tree 至少有两个 degree-1 leaves：取最长 path，其任一端若还有路外邻居就可延长，若连接路内其他点则形成 cycle。这个 leaf argument 是 tree induction 的常用入口。
<!-- bilingual-en:start -->
Every finite tree with at least two vertices has at least two degree-one leaves. Take a longest path: an endpoint with a neighbor outside the path would extend it, while an additional neighbor on the path would create a cycle. This leaf argument is a standard entry point for induction on trees.
<!-- bilingual-en:end -->

## 生成树和最小生成树
<!-- bilingual-en:start -->
*Spanning trees and minimum spanning trees*
<!-- bilingual-en:end -->

spanning subgraph 保留原图全部顶点，只删边；若它是 tree，就是 spanning tree。一张 finite undirected graph 有 spanning tree 当且仅当它 connected。构造方法是从 connected graph 反复删除 cycle 上的一条边：删 cycle edge 不破坏连通，有限步后得到 connected acyclic graph。
<!-- bilingual-en:start -->
A spanning subgraph retains every vertex of the original graph and only removes edges. If it is a tree, it is a spanning tree. A finite undirected graph has a spanning tree if and only if it is connected. Construct one by repeatedly deleting an edge on a cycle: removing a cycle edge preserves connectivity, and the finite process ends at a connected acyclic graph.
<!-- bilingual-en:end -->

BFS/DFS parent edges 也会在其可达 component 上形成 spanning tree，但“搜索树”与“最小生成树”不是同一概念：BFS tree 最小化从根出发的无权 path 长度，MST 最小化整棵 tree 的 edge-weight 总和，DFS tree 通常两者都不最小。
<!-- bilingual-en:start -->
BFS or DFS parent edges also form a spanning tree of the reachable component, but a search tree is not the same as a minimum spanning tree. A BFS tree minimizes unweighted root-to-vertex path lengths; an MST minimizes the total edge weight of the whole tree; a DFS tree generally minimizes neither.
<!-- bilingual-en:end -->

在 connected weighted graph 中，minimum spanning tree（MST）最小化
<!-- bilingual-en:start -->
In a connected weighted graph, a minimum spanning tree minimizes
<!-- bilingual-en:end -->

$$
w(T)=\sum_{e\in T}w(e).
$$

核心证明工具是 cut property：对任意非平凡 cut $(S,V\setminus S)$，跨越该 cut 的一条 minimum-weight edge 属于某个 MST；若它是唯一最轻 crossing edge，则属于每个 MST。证明用 exchange：向一个不含该边的 MST 加边产生唯一 cycle，cycle 必另有一条 crossing edge，换掉后总权重不增。
<!-- bilingual-en:start -->
The central proof tool is the cut property. For any nontrivial cut $(S,V\setminus S)$, a minimum-weight edge crossing that cut belongs to some MST; if it is the unique lightest crossing edge, it belongs to every MST. The exchange proof adds that edge to an MST that lacks it, creating one cycle; the cycle contains another crossing edge, and swapping the two does not increase total weight.
<!-- bilingual-en:end -->

Kruskal 按全局权重递增查看边，只接受连接当前两个不同 forest components 的边；Prim 从一个根生长，每次接受跨越“已进树/未进树” cut 的最轻边。两者每次选择都可由 cut property 证明 safe，最终接受 $n-1$ 条边。若所有 edge weights 互异，MST 唯一；反向不成立，存在重复权重的图也可能恰有一个 MST。
<!-- bilingual-en:start -->
Kruskal examines edges in global nondecreasing weight order and accepts only an edge joining two different forest components. Prim grows from a root and repeatedly accepts the lightest edge crossing the cut between vertices already in the tree and those outside. The cut property proves each accepted edge safe, and both algorithms finish after accepting $n-1$ edges. Distinct edge weights guarantee a unique MST; the converse is false because some graphs with tied weights still have a unique MST.
<!-- bilingual-en:end -->

> [!source] 无向分支核验
> MIT 6.042J Section 12.11 逐项给出 tree 的等价性质、forest 的 component/edge 计数、connected graph 存在 spanning tree，以及 gray-edge/cut exchange 对 MST 和 Prim/Kruskal 的证明。
> <!-- bilingual-en:start -->
> MIT 6.042J Section 12.11 establishes the equivalent tree properties, the component/edge count for forests, the existence of a spanning tree in every connected graph, and the gray-edge or cut-exchange proof for MSTs and Prim/Kruskal.
> <!-- bilingual-en:end -->

## DAG 与拓扑排序
<!-- bilingual-en:start -->
*DAGs and topological sorting*
<!-- bilingual-en:end -->

directed acyclic graph（DAG）是没有 directed cycle 的 digraph。topological order 是顶点的线性排列，使每条边 $u\to v$ 都把 $u$ 放在 $v$ 之前。对 finite digraph，以下等价：它是 DAG；它存在 topological order；每个非空 induced subgraph 至少有一个 indegree-0 source（也至少有一个 outdegree-0 sink）。
<!-- bilingual-en:start -->
A directed acyclic graph (DAG) is a digraph with no directed cycle. A topological order is a linear ordering of the vertices in which every edge $u\to v$ places $u$ before $v$. For a finite digraph, the following are equivalent: it is a DAG; it has a topological order; every nonempty induced subgraph has an in-degree-zero source and an out-degree-zero sink.
<!-- bilingual-en:end -->

source 存在性可用反证：若每点都有 incoming edge，从任一点不断沿入边逆行；有限顶点保证某点重复，重复段给 directed cycle。Kahn algorithm 因而可以反复输出任意 source，并删除它及 outgoing edges。若在输出全部顶点前没有 source，剩余子图必含 cycle；这不仅失败，还给出“依赖无法排完”的诊断。
<!-- bilingual-en:start -->
Source existence follows by contradiction. If every vertex has an incoming edge, repeatedly follow incoming edges backward. Finiteness forces a repeated vertex, producing a directed cycle. Kahn's algorithm can therefore repeatedly output any source and delete it with its outgoing edges. If no source remains before every vertex is output, the remaining subgraph contains a cycle, diagnosing why the dependencies cannot be ordered.
<!-- bilingual-en:end -->

full DFS 给另一算法：若发现指向灰色祖先的 back edge，就报告 cycle；若没有 back edge，按 finishing time 的逆序排列顶点。对每条 $u\to v$，DAG 中 $v$ 必在 $u$ 完成前完成，故逆 finishing order 把 $u$ 放在 $v$ 前。
<!-- bilingual-en:start -->
Full DFS gives another algorithm. A back edge to a gray ancestor reports a cycle; if no back edge exists, list vertices in reverse finishing-time order. For every edge $u\to v$ in a DAG, $v$ must finish before $u$, so reverse finishing order places $u$ before $v$.
<!-- bilingual-en:end -->

topological order 通常不唯一：两个互不依赖的 sources 可交换。它表示依赖约束允许的一条 linear extension，不表示图本身规定了唯一时间线。每条边都必须向前，但没有边的两点也可能因间接 path 可比较；若二者完全不可达，则可并行或在不同合法顺序中交换。
<!-- bilingual-en:start -->
A topological order is generally not unique: two independent sources can be exchanged. It is one linear extension permitted by the dependency constraints, not a unique timeline dictated by the graph. Every edge must point forward, while vertices without a direct edge may still be comparable through an indirect path. Mutually unreachable tasks can be parallelized or exchanged across valid orders.
<!-- bilingual-en:end -->

若每个任务耗时 1 且处理器无限，把顶点按“以它结尾的最长 chain 长度”分层，可在最长 chain 的顶点数这么多步内完成；任何 chain 内任务又必须逐期执行，因此该 critical-path 下界可达。处理器有限或任务时长不同，还要同时考虑总工作量和资源约束，topological order 本身不等于最优 schedule。
<!-- bilingual-en:start -->
If every task takes one unit of time and processors are unlimited, layering vertices by the length of a longest chain ending at each vertex completes the work in as many steps as the largest chain has vertices. Tasks on any chain must be sequential, so this critical-path lower bound is attainable. With limited processors or unequal durations, total work and resource constraints also matter; a topological order alone is not an optimal schedule.
<!-- bilingual-en:end -->

> [!source] 有向分支核验
> MIT 6.042J Section 10.5 由 prerequisite graph 引入 DAG，证明 finite DAG 存在 topological sort，并把最长 chain 与无限处理器 schedule 联系起来；MIT 6.006 Lecture 10 进一步核验 DFS reverse finishing order 与 back-edge cycle certificate。
> <!-- bilingual-en:start -->
> MIT 6.042J Section 10.5 introduces DAGs through prerequisite graphs, proves that every finite DAG has a topological sort, and connects longest chains with unlimited-processor schedules. MIT 6.006 Lecture 10 additionally supports reverse DFS finishing order and the back-edge cycle certificate.
> <!-- bilingual-en:end -->

## 完整例子一：先修关系是否可完成
<!-- bilingual-en:start -->
*Worked example one: can the prerequisites be completed?*
<!-- bilingual-en:end -->

设边为 $A\to C,B\to C,C\to D,B\to E$。初始 sources 是 $A,B$。若 Kahn algorithm 先输出 $A$ 再输出 $B$，删除相关边后 $C,E$ 成为 sources；一个合法顺序是 $A,B,C,E,D$，另一个是 $B,A,E,C,D$。顺序不同，但每条 prerequisite edge 都向前。
<!-- bilingual-en:start -->
Let the edges be $A\to C,B\to C,C\to D,B\to E$. The initial sources are $A$ and $B$. If Kahn's algorithm outputs $A$ and then $B$, deleting their outgoing edges makes $C$ and $E$ sources. One valid order is $A,B,C,E,D$; another is $B,A,E,C,D$. The orders differ, but every prerequisite edge points forward.
<!-- bilingual-en:end -->

若再加 $D\to B$，则 $B\to C\to D\to B$ 成为 directed cycle。删除 $A$ 后，剩余 cycle 中没有 indegree-0 source，Kahn algorithm 提前停止；DFS 则会遇到指向当前祖先的 back edge。两种失败都给出“不是 DAG”的可核验证书，而不只是返回空结果。
<!-- bilingual-en:start -->
Adding $D\to B$ creates the directed cycle $B\to C\to D\to B$. After removing $A$, the remaining cycle has no in-degree-zero source, so Kahn's algorithm stops early. DFS instead encounters a back edge to an active ancestor. Both failures provide a checkable certificate that the graph is not a DAG rather than merely returning an empty result.
<!-- bilingual-en:end -->

## 完整例子二：用 cut property 找 MST
<!-- bilingual-en:start -->
*Worked example two: finding an MST with the cut property*
<!-- bilingual-en:end -->

设 weighted graph 的边为 $AB=1,BC=2,CD=3,AC=4,BD=5$。Kruskal 依次接受 $AB,BC,CD$，此时四点 connected 且有 $4-1=3$ 条边，所以得到 tree，总权重 6。$AC$ 会在 $A-B-C-A$ 上成环，$BD$ 更重且无需使用。
<!-- bilingual-en:start -->
Let a weighted graph have edges $AB=1,BC=2,CD=3,AC=4,BD=5$. Kruskal accepts $AB$, $BC$, and $CD$ in order. The four vertices are then connected with $4-1=3$ edges, so the result is a tree of total weight six. Edge $AC$ would close the cycle $A-B-C-A$, while $BD$ is heavier and unnecessary.
<!-- bilingual-en:end -->

安全性可以逐 cut 查看：cut $\{A\}|\{B,C,D\}$ 的唯一最轻 crossing edge 是 $AB$；加入后，cut $\{A,B\}|\{C,D\}$ 的唯一最轻边是 $BC$；再看 $\{A,B,C\}|\{D\}$，唯一最轻边是 $CD$。三条边因此属于每个 MST；本例权重互异，MST 唯一。
<!-- bilingual-en:start -->
Safety can be checked cut by cut. Across $\{A\}|\{B,C,D\}$, $AB$ is the unique lightest crossing edge. After adding it, $BC$ is uniquely lightest across $\{A,B\}|\{C,D\}$. Finally, $CD$ is uniquely lightest across $\{A,B,C\}|\{D\}$. All three edges therefore belong to every MST; because all weights are distinct here, the MST is unique.
<!-- bilingual-en:end -->

## 诊断与边界
<!-- bilingual-en:start -->
*Diagnostics and boundaries*
<!-- bilingual-en:end -->

| 常见说法 | 缺了什么 | 正确修复 |
|---|---|---|
| “有 $n-1$ 条边，所以是 tree” | connected 或 acyclic | 再验证其中一个结构条件 |
| “DAG 所以两点间 unique path” | DAG 只排除 directed cycle | unique path 是 tree 性质，不是一般 DAG 性质 |
| “BFS tree 就是 MST” | 两个优化目标不同 | BFS 最少边数；MST 最小总权重 |
| “边权有并列，所以 MST 不唯一” | 并列只是可能性 | 构造两棵等权 trees 才能证明不唯一 |
| “有一个 topological order，所以任意顺序都行” | 仍须尊重全部 edges | 每条 $u\to v$ 必须把 $u$ 放前 |
| “topological order 是最优调度” | 未考虑时长、处理器和资源 | 另做 scheduling optimization |
<!-- bilingual-en:start -->
| Common claim | What is missing | Correct repair |
|---|---|---|
| “It has $n-1$ edges, so it is a tree.” | Connectivity or acyclicity | Verify one additional structural condition |
| “It is a DAG, so every pair has a unique path.” | A DAG only excludes directed cycles | Unique paths characterize trees, not general DAGs |
| “A BFS tree is an MST.” | The objectives differ | BFS minimizes edge count; an MST minimizes total weight |
| “Some weights tie, so the MST is not unique.” | Ties create only a possibility | Exhibit two equal-weight spanning trees to prove nonuniqueness |
| “One topological order exists, so any order works.” | Every edge still constrains the order | Place every $u$ before $v$ for each $u\to v$ |
| “A topological order is an optimal schedule.” | Durations, processors, and resources are omitted | Solve the additional scheduling problem |
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 用自己的话解释：为什么 tree 中任意两点只有一条 simple path？
<!-- bilingual-en:start -->
*Explain in your own words: why is there exactly one simple path between any two vertices in a tree?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> connected 保证至少有一条。若有两条不同 paths，它们在分叉与重会之间形成 cycle，与 tree 的 acyclic 条件矛盾。
> <!-- bilingual-en:start -->
> Connectivity guarantees at least one path. If two distinct paths existed, their segments between a divergence and reunion would form a cycle, contradicting acyclicity.
> <!-- bilingual-en:end -->

### 一个 20 顶点 forest 有 4 个 components，应有多少条边？
<!-- bilingual-en:start -->
*How many edges does a forest with 20 vertices and four components have?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> $20-4=16$。每个 component 是 tree，分别有 $n_i-1$ 条边，求和得到 $n-c$。
> <!-- bilingual-en:start -->
> $20-4=16$. Each component is a tree with $n_i-1$ edges, and summing gives $n-c$.
> <!-- bilingual-en:end -->

### 为什么 cut 上“唯一最轻边”必须出现在每个 MST 中？
<!-- bilingual-en:start -->
*Why must the unique lightest edge across a cut appear in every MST?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 若某 MST 不含它，加边会产生 cycle；cycle 还有另一条跨 cut 的边。用唯一更轻边换掉那条边会严格降低总权重，和原 tree 最小矛盾。
> <!-- bilingual-en:start -->
> If an MST omitted it, adding the edge would create a cycle containing another edge across the cut. Replacing that edge by the uniquely lighter one would strictly reduce total weight, contradicting minimality.
> <!-- bilingual-en:end -->

### Kahn algorithm 在输出全部顶点前没有 source，说明什么？
<!-- bilingual-en:start -->
*What does it mean if Kahn's algorithm has no source before all vertices are output?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 剩余非空 induced subgraph 每点都有入边；沿入边逆行必因有限性重复顶点，得到 directed cycle。因此原图不是 DAG。
> <!-- bilingual-en:start -->
> Every vertex in the remaining nonempty induced subgraph has an incoming edge. Following incoming edges backward must repeat a vertex by finiteness, producing a directed cycle. The original graph is not a DAG.
> <!-- bilingual-en:end -->

### 为什么 DAG 可以有多条从 $A$ 到 $D$ 的 paths，却仍然 acyclic？
<!-- bilingual-en:start -->
*Why can a DAG contain several paths from $A$ to $D$ and still be acyclic?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> acyclic 只禁止沿方向回到起点，不禁止分叉后在下游汇合。$A\to B\to D$ 与 $A\to C\to D$ 可同时存在，只要没有从 $D$ 回到上游的 directed path。
> <!-- bilingual-en:start -->
> Acyclicity only forbids returning to the starting vertex along edge directions; it does not forbid branches from merging downstream. Both $A\to B\to D$ and $A\to C\to D$ may exist provided there is no directed path from $D$ back upstream.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf]]，Section 10.5：核验 DAG、topological sort、minimal/source 构造、chain/critical path 与无限处理器 scheduling。
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf]]，Section 12.11：核验 tree 等价刻画、leaves、forest 的 $n-c$ 边、spanning tree、MST、gray-edge/cut exchange 和 Prim/Kruskal。
- MIT OpenCourseWare 6.006, [Lecture 10: Depth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf)：核验 DAG iff topological order、reverse finishing order 与 back-edge cycle detection。
- [[01_Math/07-Mathematics for Computer Science/02_Structures.md#Session 17 — Directed Acyclic Graphs]] 与 [[01_Math/07-Mathematics for Computer Science/02_Structures.md#Session 21 — Trees and Minimum Spanning Trees]]：支持课程例题与术语；核心定理和算法已对照教材/OCW 讲义复核。
<!-- bilingual-en:start -->
- Section 10.5 of [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] supports DAGs, topological sorting, source-based construction, chains, critical paths, and unlimited-processor scheduling.
- Section 12.11 of [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] supports equivalent tree characterizations, leaves, the $n-c$ forest edge count, spanning trees, MSTs, gray-edge or cut exchange, and Prim/Kruskal.
- MIT OpenCourseWare 6.006, [Lecture 10: Depth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf), supports the equivalence between DAGs and topological order, reverse finishing order, and back-edge cycle detection.
- The local course sections on [[01_Math/07-Mathematics for Computer Science/02_Structures.md#Session 17 — Directed Acyclic Graphs|directed acyclic graphs]] and [[01_Math/07-Mathematics for Computer Science/02_Structures.md#Session 21 — Trees and Minimum Spanning Trees|trees and minimum spanning trees]] support course examples and terminology. Core theorems and algorithms were rechecked against the textbook and OCW lecture.
<!-- bilingual-en:end -->

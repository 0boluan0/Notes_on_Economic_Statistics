---
aliases:
  - Depth-first search reachability
  - DFS 单源可达与非最短父路径
student_os: knowledge-atom
atom_id: MCS-GRAPH-016
atom_type: procedure
status: source-checked
part_of:
  - "[[图的基本结构、路径与遍历.canvas]]"
requires:
  - "[[搜索标记与父森林]]"
  - "[[图距离与可达性]]"
contrasts_with:
  - "[[BFS 分层与无权最短路]]"
leads_to:
  - "[[有向 DFS 的环证书]]"
  - "[[全图搜索与强连通边界]]"
---

<!-- bilingual-en:start -->
# Depth-first search
<!-- bilingual-en:end -->

*DFS：可达性与非最短父路径*

<!-- bilingual-en:start -->
Depth-first search, or **DFS**, explores one branch before returning to a choice it postponed. When it first reaches a new vertex, it explores that vertex's neighbors completely before resuming the previous vertex. This nested pattern produces a route to every reachable vertex, but it does not choose routes in order of their length.
<!-- bilingual-en:end -->

**深度优先搜索**（DFS）先探索一条分支，再返回先前暂缓的选择。第一次到达新顶点时，会先完成对该顶点邻居的探索，然后才继续原顶点的工作。这种嵌套顺序能为每个可达点找出一条路线，但不会按路线长度从小到大选择。

<!-- bilingual-en:start -->
Use a finite [[图类型的建模边界|graph]], choose an existing source vertex $s$, and initially leave every vertex undiscovered. In a directed graph, inspect outgoing edges; in an undirected graph, either direction along an edge is allowed. An [[邻接表与邻接矩阵|adjacency list]] specifies the neighbors and the order in which they are inspected. A vertex is [[图距离与可达性|reachable]] if a permitted path from $s$ ends there.
<!-- bilingual-en:end -->

取一张有限[[图类型的建模边界|图]]，选择其中的源点 $s$，开始时所有顶点均未发现。有向图检查出边，无向边则允许双向行走。[[邻接表与邻接矩阵|邻接表]]给出邻居及其检查次序；若存在从 $s$ 出发、遵守方向的路径到达某点，该点就是[[图距离与可达性|可达]]的。

<!-- bilingual-en:start -->
## Explore a child, then resume its parent
<!-- bilingual-en:end -->

*先探索子节点，再恢复父节点*

<!-- bilingual-en:start -->
Mark $s$ discovered and give it no parent. The procedure `visit(u)` scans the neighbors of an already discovered vertex $u$. If the next neighbor $v$ is undiscovered, immediately mark it, record $u$ and the chosen edge as its parent information, and call `visit(v)`. Calling the same procedure inside itself is **recursion**: the scan at $u$ is suspended while the scan at $v$ runs. Only when `visit(v)` finishes does $u$ resume at its next adjacency entry. An already discovered neighbor starts no new call. After the entire list has been scanned, `visit(u)` finishes and returns.
<!-- bilingual-en:end -->

先标记 $s$ 已发现，并设为无父顶点。过程 `visit(u)` 扫描一个已发现顶点 $u$ 的邻居。若下一个邻居 $v$ 尚未发现，就立即标记，记录父顶点 $u$ 和所用的边，再调用 `visit(v)`。在过程中再次调用自身就是**递归**：$u$ 的扫描暂停，先执行 $v$ 的扫描；只有 `visit(v)` 完成后，$u$ 才从下一个邻接项继续。已发现的邻居不会引发新调用。整张邻接表扫描完毕，`visit(u)` 才完成并返回。

<!-- bilingual-en:start -->
This is the [[搜索标记与父森林|first-discovery parent rule]] with a particular exploration order. A self-loop cannot rediscover its own endpoint, and parallel edges do not cause a vertex to be entered repeatedly. In a multigraph, retain the identity of the edge actually used to discover a child. An isolated source is discovered and immediately finishes. The empty graph has no source to select.
<!-- bilingual-en:end -->

这是在[[搜索标记与父森林|首次发现时确定父记录]]的规则上，指定一种探索顺序。自环不会重新发现自己的端点，平行边也不会使同一顶点反复进入递归。多重图应保留真正用于发现子节点的那条边的身份。孤立源点会被发现并立即完成；空图则没有可选的源点。

> [!example] A long parent route beside a direct edge · 直达边旁的较长父路径
> <!-- bilingual-en:start -->
> Take $s\to a$, $a\to b$, $b\to v$, and $s\to v$, with $a$ listed before $v$ at $s$. DFS discovers $s,a,b,v$ in that order. Its parent records are $\operatorname{parent}[a]=s$, $\operatorname{parent}[b]=a$, and $\operatorname{parent}[v]=b$. It then finishes $v,b,a$, resumes $s$, and sees that $v$ is already discovered. The parent route $s,a,b,v$ has three edges, although the direct route $s,v$ has one. Changing the order at $s$ can change the parent route.
> <!-- bilingual-en:end -->
>
> 取边 $s\to a$、$a\to b$、$b\to v$、$s\to v$，并把 $s$ 的邻居 $a$ 排在 $v$ 前。DFS 按 $s,a,b,v$ 发现顶点，父记录为 $\operatorname{parent}[a]=s$、$\operatorname{parent}[b]=a$、$\operatorname{parent}[v]=b$。随后依次完成 $v,b,a$，回到 $s$ 时才检查直达边，但 $v$ 已发现。父路径 $s,a,b,v$ 有三条边，而直达路径 $s,v$ 只有一条；改变 $s$ 的邻接次序可能改变父路径。

<!-- bilingual-en:start -->
## Why every reachable vertex is found
<!-- bilingual-en:end -->

## 可达性正确性

<!-- bilingual-en:start -->
The procedure terminates: every vertex starts at most one call, every adjacency list is finite, and each suspended scan resumes from its saved position. Every discovered vertex is reachable, because its first parent edge extends an existing path from $s$. Parent chains have no repeated vertices: each step toward a parent moves to an earlier discovery.
<!-- bilingual-en:end -->

过程会终止：每个顶点至多开启一次调用，每张邻接表有限，而且暂停的扫描会从保存的位置继续。每个已发现点都可达，因为首次父边把一条从 $s$ 出发的已有路径延长了一步。父链不会重复顶点，因为每次走向父顶点都会退回更早的一次发现。

<!-- bilingual-en:start -->
Conversely, suppose a [[游走路径与环|path]] from $s$ ends at a vertex still undiscovered when the search finishes. Along that path, take the first undiscovered vertex $v$ and its preceding vertex $u$. The source was marked, so this predecessor exists and was discovered. Before finishing, `visit(u)` must inspect the edge to $v$; it would discover $v$ if no earlier call had done so. This contradicts the choice of $v$. Thus a complete fresh source search finds **exactly** the source-reachable vertices. The path-length induction in the lecture gives the same conclusion.
<!-- bilingual-en:end -->

反过来，假设搜索结束后，某条从 $s$ 出发的[[游走路径与环|路径]]仍以未发现点结束。沿路径取第一个未发现点 $v$，及其前一个点 $u$。源点早已标记，因此这个前驱存在且已发现。`visit(u)` 完成前必检查通向 $v$ 的边；若之前还没发现 $v$，此时就会发现，矛盾。因此，一次完整的新单源搜索**恰好**找到源点可达的顶点。讲义按路径长度归纳，也得到同一结论。

<!-- bilingual-en:start -->
## Why a valid route need not be shortest
<!-- bilingual-en:end -->

## 为什么 parent path 不承担最短语义

<!-- bilingual-en:start -->
The three-edge route in the example is a correct reachability certificate: all its edges exist and have the required directions. Calling it a shortest route would be the error. DFS commits to the first branch and does not revise a parent merely because a later edge offers a shorter route. For minimum **edge count** in an unweighted graph, [[BFS 分层与无权最短路|BFS]] uses FIFO order to discover vertices by distance layers. DFS may happen to return a shortest path on a particular input, but provides no general guarantee.
<!-- bilingual-en:end -->

例中的三边路线是正确的可达性证书：各条边真实存在，方向也合法。错误只在于把它称为最短路线。DFS 先深入首次选择的分支，不会因为后来出现较短路线而重写父记录。无权图若要最少的**边数**，[[BFS 分层与无权最短路|BFS]] 使用 FIFO 次序按距离层发现顶点。某个具体输入上，DFS 可能恰好返回最短路径，但它没有普遍保证。

<!-- bilingual-en:start -->
## What an iterative DFS must remember
<!-- bilingual-en:end -->

*迭代 DFS 必须记住什么*

<!-- bilingual-en:start -->
An iterative version can reproduce recursion with a stack of **frames**. Each frame records a vertex and the next adjacency entry to inspect. When a new child is discovered, push its frame and suspend the old one; when its list ends, pop its frame and resume the parent. At any edge inspection, the unfinished frames form one root-to-current-vertex parent chain. These are the active ancestors used by the [[有向 DFS 的环证书|directed-cycle certificate]].
<!-- bilingual-en:end -->

迭代版本可以用**栈帧**复现递归：每个栈帧记录顶点，以及下一个待检查的邻接项。发现新子节点时，压入其栈帧并暂停旧帧；子节点列表结束时弹出，再恢复父节点。检查任何一条边时，未完成的栈帧都构成一条从根到当前顶点的父链；[[有向 DFS 的环证书|有向环证书]]使用的正是这些活动祖先。

<!-- bilingual-en:start -->
Merely replacing a queue with a last-in, first-out container and pushing all neighbors in advance does not preserve this structure. For example, pushing both children $a,b$ of $s$ leaves a sibling waiting while the other is processed; that waiting sibling is not an ancestor. Such a worklist can still find reachable vertices with suitable marking, but the recursion-based ancestor and finishing-order arguments cannot be transferred automatically. [[全图搜索与强连通边界|Full traversal]] restarts at undiscovered vertices while retaining marks; its component interpretation is a separate question. The [[邻接表图搜索复杂度|running-time bound]] likewise states its own representation and operation assumptions.
<!-- bilingual-en:end -->

仅把队列换成后进先出容器，再提前压入所有邻居，并不保留这种结构。例如，把 $s$ 的两个子节点 $a,b$ 都压入后，处理其中一个时，另一个兄弟节点仍在等待；它并不是祖先。这类工作表在适当标记下仍可寻找可达点，但不能自动套用递归的祖先与完成次序论证。[[全图搜索与强连通边界|全图遍历]]保留标记并从未发现点重启，其分量含义另行判断；[[邻接表图搜索复杂度|运行时间界]]也另有表示法和操作成本前提。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.006, Lecture 10: Depth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf), PDF pp. 2–4: recursive procedure, reachability proof, non-shortest parent paths, full traversal and active ancestors. The stack-frame explanation spells out the suspended-scan behavior of this procedure; the source's simple-graph convention is extended here only where first-discovery marking gives the stated loop/parallel-edge behavior.
<!-- bilingual-en:end -->

- MIT 6.006 Lecture 10，第 2–4 页：递归过程、可达性证明、父路径不保证最短、全图遍历及活动祖先。栈帧段展开了该过程的暂停扫描行为；讲义的简单图约定仅在首次发现标记可直接保证之处，扩展到所述自环/平行边情形。

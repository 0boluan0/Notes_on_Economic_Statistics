---
aliases:
- Reverse DFS finishing order topologically sorts a DAG
- DFS 逆完成顺序拓扑排序
student_os: knowledge-atom
atom_id: MCS-GRAPH-035
atom_set: acyclic-graphs-trees-dags
atom_type: algorithm
status: source-checked
requires:
- '[[DAG 与拓扑序等价]]'
- '[[有向 DFS 的环证书]]'
- '[[全图搜索与强连通边界]]'
related:
- '[[Kahn 拓扑排序与环证书]]'
part_of:
- '[[无环图：树、生成树、DAG 与拓扑排序.canvas]]'
---

<!-- bilingual-en:start -->
# Topological sorting by DFS
<!-- bilingual-en:end -->

*DFS 逆完成序拓扑排序*

<!-- bilingual-en:start -->
When depth-first search (DFS) finishes a vertex, it has finished exploring every new branch started from that vertex. In a graph without directed cycles, this places each arrow's destination earlier in the **finishing** list than its origin. Reversing the complete finishing list therefore puts prerequisites before their dependants. The list we reverse records completed visits, not first discoveries.
<!-- bilingual-en:end -->

DFS 完成一个顶点时，从它开启的新分支也都已探索完。在没有有向环的图中，每条箭头的终点会比起点更早进入**完成**列表。因此，把完整的完成列表反转，就能让先修项排在依赖它的项目前面。这里反转的是访问完成次序，不是首次发现次序。

<!-- bilingual-en:start -->
Take a finite [[图类型的建模边界|directed graph]] $G=(V,E)$, with vertices $V$ and arrows $E$. A **[[有向无环图|DAG]]** has no directed [[游走路径与环|cycle]]. A **[[拓扑序|topological order]]** is a list containing every vertex exactly once, with $u$ before $v$ for every edge $u\to v$. The [[DAG 与拓扑序等价|existence theorem]] says that precisely the DAGs admit such an order. This card constructs one using [[DFS 可达性与非最短路径|standard recursive DFS]], or equivalent stack frames that remember where each suspended adjacency scan will resume.
<!-- bilingual-en:end -->

取有限[[图类型的建模边界|有向图]] $G=(V,E)$，其中 $V$ 为顶点集、$E$ 为箭头集合。**[[有向无环图|DAG]]** 没有有向[[游走路径与环|环]]；**[[拓扑序]]**让每个顶点恰好出现一次，并对每条边 $u\to v$ 把 $u$ 放在 $v$ 前。[[DAG 与拓扑序等价|存在性定理]]说明，恰好是 DAG 能有这样的顺序。本卡用[[DFS 可达性与非最短路径|标准递归 DFS]]构造它；等价的栈帧实现必须保存每次暂停的邻接扫描位置。

<!-- bilingual-en:start -->
## Append a vertex only when its visit finishes
<!-- bilingual-en:end -->

*访问真正完成时才写入列表*

<!-- bilingual-en:start -->
Initially every vertex is undiscovered. Run a [[全图搜索与强连通边界|full DFS]]: choose an undiscovered root, finish its entire recursive exploration, and repeat at another undiscovered root while retaining all marks. Within a visit to $u$, inspect each outgoing neighbor; recursively visit it only if it is still undiscovered. Append $u$ to a list **after** its whole adjacency scan and all child visits have finished. Finally reverse that list.
<!-- bilingual-en:end -->

起初全部顶点均未发现。进行[[全图搜索与强连通边界|完整 DFS]]：选未发现点作根，完成它的整个递归探索，再从另一个未发现点重启，保留全部旧标记。访问 $u$ 时逐个检查出邻居，仅对尚未发现的邻居递归调用；必须等整张邻接表扫描和全部子调用都结束后，才把 $u$ 加入列表。最后反转这个列表。

<!-- bilingual-en:start -->
Every vertex is discovered once and finishes once. Thus a full run gives a permutation of all vertices, even when the graph is disconnected or has isolated vertices. If there are no vertices, the finishing list and its reverse are both empty. These coverage facts alone do not yet prove that every edge points forward; that is where acyclicity enters.
<!-- bilingual-en:end -->

每个顶点只发现一次、完成一次，因此完整运行会得到全部顶点的一个排列，即使图不连通或含有孤立点也一样。没有顶点时，完成列表及其逆序都为空。仅有覆盖性还没有证明每条边都指向后面；接下来才用到无环条件。

> [!example] Discovery and completion are different lists · 发现列表与完成列表不同
> <!-- bilingual-en:start -->
> Use edges $a\to b$, $a\to c$, $b\to d$, $c\to d$, and an isolated vertex $e$. Start at $a$, inspect $b$ before $c$, and process $e$ after that tree. The discovery list is $a,b,d,c,e$. Vertex $d$ finishes first, then $b$, then $c$, then $a$, and finally $e$: the finishing list is $d,b,c,a,e$. Its reverse $e,a,c,b,d$ is topological. The discovery list is not: it puts $d$ before $c$ despite $c\to d$.
> <!-- bilingual-en:end -->
>
> 取边 $a\to b$、$a\to c$、$b\to d$、$c\to d$，另加孤立顶点 $e$。先从 $a$ 出发，先检查 $b$ 后检查 $c$，整棵树完成后再处理 $e$。发现列表是 $a,b,d,c,e$；最先完成的是 $d$，之后依次为 $b,c,a,e$，所以完成列表为 $d,b,c,a,e$。逆序 $e,a,c,b,d$ 是拓扑序；发现列表却不是，因为边 $c\to d$ 要求 $c$ 在 $d$ 前。

<!-- bilingual-en:start -->
## Check one edge while its tail is active
<!-- bilingual-en:end -->

*在起点调用活动时检查一条边*

<!-- bilingual-en:start -->
Let $f(u)$ be the position of $u$ in the finishing list; it is a counter for completion order, not a physical time. We will prove that every edge $u\to v$ in a DAG satisfies $f(v)<f(u)$. At the moment DFS inspects this edge, the call for $u$ has not yet finished. Use the [[有向 DFS 的环证书|three DFS states]]: white means undiscovered, gray means its recursive call is active, and black means finished.
<!-- bilingual-en:end -->

令 $f(u)$ 为 $u$ 在完成列表中的位置；它是完成次序计数，不是物理时间。要证明 DAG 中每条边 $u\to v$ 都满足 $f(v)<f(u)$。DFS 检查这条边时，$u$ 的调用尚未完成。使用[[有向 DFS 的环证书|三种 DFS 状态]]：白色表示未发现，灰色表示递归调用仍活动，黑色表示已经完成。

<!-- bilingual-en:start -->
If $v$ is white, DFS starts its child visit immediately and waits for that visit to finish before resuming $u$. Hence $v$ finishes before $u$. If $v$ is black, it has already finished, whereas $u$ has not, so the same inequality holds.
<!-- bilingual-en:end -->

若 $v$ 为白色，DFS 立刻开启它的子调用，并等该调用完成后才恢复 $u$，因此 $v$ 比 $u$ 先完成。若 $v$ 为黑色，它已经完成，而 $u$ 尚未完成，也得到同一不等式。

<!-- bilingual-en:start -->
The remaining possibility is gray. In recursive DFS, the active calls are exactly one root-to-current-vertex parent chain, so a gray $v$ is $u$ itself or an ancestor of $u$. The directed parent path from $v$ to $u$, followed by $u\to v$, would form a cycle. A DAG rules this out, including the self-loop case $v=u$. These three cases exhaust the possibilities, proving $f(v)<f(u)$ for each edge.
<!-- bilingual-en:end -->

剩下的可能是灰色。递归 DFS 的活动调用恰好形成从根到当前顶点的一条父链，所以灰色的 $v$ 是 $u$ 本身或它的祖先。从 $v$ 沿有向父路径走到 $u$，再走 $u\to v$，就形成有向环。DAG 排除了这种情况，也包括 $v=u$ 的自环。三种状态已经穷尽，因此对每条边都有 $f(v)<f(u)$。

<!-- bilingual-en:start -->
Write $n=|V|$ for the number of vertices. Reversal gives a vertex $u$ the new position $n+1-f(u)$. Thus $f(v)<f(u)$ becomes $n+1-f(u)<n+1-f(v)$: $u$ precedes $v$, exactly the topological-order condition. The argument also covers edges pointing into an earlier finished DFS tree, since their destination is black.
<!-- bilingual-en:end -->

记顶点数为 $n=|V|$。反转后，顶点 $u$ 的新位置为 $n+1-f(u)$。于是 $f(v)<f(u)$ 变成 $n+1-f(u)<n+1-f(v)$，也就是 $u$ 位于 $v$ 之前，恰好满足拓扑序条件。指向更早完成的 DFS 树的边也已经包含在黑色终点情形中。

<!-- bilingual-en:start -->
## Reject cycles and include every root
<!-- bilingual-en:end -->

*有环时拒绝，遍历时覆盖所有根*

<!-- bilingual-en:start -->
If the input is not promised to be a DAG, check for a gray-target edge while running DFS. Such an edge and the parent chain give a [[有向 DFS 的环证书|directed-cycle certificate]], so reject topological sorting. A full recursive DFS finds such an edge whenever a directed cycle exists. If one instead completes DFS and mechanically reverses the list on a cyclic graph, the result is still a permutation, but at least one edge violates its ordering. A permutation is not itself a correctness certificate.
<!-- bilingual-en:end -->

若输入没有保证是 DAG，就在 DFS 中检查是否遇到灰色终点。这样的边及父链给出[[有向 DFS 的环证书|有向环证书]]，应拒绝输出拓扑序。只要图中有有向环，完整递归 DFS 就会发现这种边。若仍在有环图上跑完 DFS 并机械反转列表，结果虽然是一个排列，却至少违反一条边的先后要求。一个排列本身不是正确性证书。

> [!question]- Why is a single source insufficient? · 为什么单源搜索还不够？
> <!-- bilingual-en:start -->
> In the displayed example, starting only from $a$ never visits the isolated vertex $e$. Reversing that partial finishing list cannot be a topological order of the whole graph because it omits a vertex. Full DFS supplies the missing roots; resetting all marks between roots can instead rediscover old vertices, so it no longer guarantees one completion per vertex.
> <!-- bilingual-en:end -->
>
> 在展示的例子中，只从 $a$ 搜索永远访问不到孤立顶点 $e$。反转这份部分完成列表不可能是全图的拓扑序，因为它遗漏了一个顶点。Full DFS 补齐其余根；若每换一个根就清空旧标记，则可能再次发现旧顶点，不再保证每点只完成一次。

<!-- bilingual-en:start -->
The proof depends on nested visits. A last-in, first-out work list that merely pushes all neighbors need not have the active-ancestor structure, so its processing order cannot simply replace recursive finishing order. With [[邻接表图搜索复杂度|adjacency-list and constant-time-operation assumptions]], full DFS, one append per vertex and one final reversal take $O(|V|+|E|)$ time on nonempty inputs and $O(|V|)$ auxiliary space. Neighbor and root choices may change the resulting legal order. [[Kahn 拓扑排序与环证书|Kahn's algorithm]] constructs an order using remaining in-degrees instead.
<!-- bilingual-en:end -->

证明依赖调用的嵌套关系。仅把全部邻居压入后进先出工作列表，不一定保留活动祖先结构，不能直接拿它的处理顺序替代递归完成序。满足[[邻接表图搜索复杂度|邻接表及常数时间操作前提]]时，完整 DFS、每点追加一次与最后一次反转，对非空输入共需 $O(|V|+|E|)$ 时间和 $O(|V|)$ 辅助空间。邻接顺序与根的选择可能改变所得合法排列；[[Kahn 拓扑排序与环证书|Kahn 算法]]则用剩余入度构造顺序。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.006 Lecture 10: Depth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf), PDF2–4: recursive visit, full DFS, finishing order, the per-edge reverse-order argument, and cycle detection using active ancestors. The proof above uses the three states at edge inspection to make the nesting conditions explicit.
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=441|MIT Mathematics for Computer Science (2018), Definition 10.5.2, PDF441]]: a topological list contains all vertices and respects the prerequisite directions.
<!-- bilingual-en:end -->

- MIT 6.006 Lecture 10，PDF2–4：递归访问、完整 DFS、完成序、逐边逆序证明，以及活动祖先环检测。正文在检查边的时刻分三种状态，明确使用递归嵌套条件。
- MIT《Mathematics for Computer Science》（2018）定义10.5.2，PDF441：拓扑列表包含全部顶点，并遵守先修方向。

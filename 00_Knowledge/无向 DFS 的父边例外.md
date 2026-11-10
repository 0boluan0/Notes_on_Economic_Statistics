---
aliases:
  - Undirected DFS parent-edge exception
  - 无向 DFS 父边判别
student_os: knowledge-atom
atom_id: MCS-GRAPH-018
atom_type: diagnostic
status: source-checked
part_of:
  - "[[图的基本结构、路径与遍历.canvas]]"
requires:
  - "[[有向 DFS 的环证书]]"
  - "[[图类型的建模边界]]"
leads_to:
  - "[[全图搜索与强连通边界]]"
---

<!-- bilingual-en:start -->
# Parent edges in undirected DFS
<!-- bilingual-en:end -->

*无向 DFS 的父边例外*

<!-- bilingual-en:start -->
An undirected edge can be read from either endpoint. After DFS uses it to discover a child, reading the same edge back toward the parent does not create a second edge or a cycle. The exception is about **which edge was used**, not merely which vertex is seen. This distinction is essential when two different edges have the same endpoints.
<!-- bilingual-en:end -->

无向边可以从任一端点读取。DFS 用它发现子节点后，再沿同一条边看到父节点，并没有产生第二条边或一个环。例外针对的是**究竟用了哪一条边**，而不只是遇见哪个顶点。两条不同边具有相同端点时，这一区别尤其重要。

<!-- bilingual-en:start -->
Consider standard recursive [[DFS 可达性与非最短路径|DFS]], or equivalent stack frames, on a finite [[图类型的建模边界|undirected graph]]. Each stored adjacency entry identifies an incident edge and its other endpoint. When $p$ discovers $u$ through edge $e$, store both $\operatorname{parent}[u]=p$ and $\operatorname{parentEdge}[u]=e$. A root has neither. As in the [[有向 DFS 的环证书|directed certificate]], gray means an active DFS call and black means a completed one.
<!-- bilingual-en:end -->

在有限[[图类型的建模边界|无向图]]上使用标准递归[[DFS 可达性与非最短路径|DFS]]或等价栈帧。每个邻接项标识一条关联边及其另一端点。若 $p$ 经边 $e$ 发现 $u$，同时记录 $\operatorname{parent}[u]=p$ 和 $\operatorname{parentEdge}[u]=e$；根两者皆无。与[[有向 DFS 的环证书|有向环证书]]一样，灰色表示活动调用，黑色表示已经完成。

> [!example] A tree is not a triangle · 树不是三角形
> <!-- bilingual-en:start -->
> In the [[无向树|tree]] $a-b-c$, DFS from $a$ uses edges $e_1=\{a,b\}$ and $e_2=\{b,c\}$. At $b$, seeing $a$ through $e_1$ is just the reverse reading of its parent edge; at $c$, the same holds for $e_2$. Neither is a cycle. Add a new edge $e_3=\{c,a\}$: at $c$, it reaches active ancestor $a$ and differs from $e_2$. The parent path $a,b,c$ plus $e_3$ now gives a triangle.
> <!-- bilingual-en:end -->
>
> 在[[无向树|树]] $a-b-c$ 中，从 $a$ 出发的 DFS 使用 $e_1=\{a,b\}$、$e_2=\{b,c\}$。$b$ 经 $e_1$ 看到 $a$，只是反向读取自己的父边；$c$ 对 $e_2$ 也是如此，均不是环。若新增 $e_3=\{c,a\}$，$c$ 会经这条不同于 $e_2$ 的边遇到活动祖先 $a$，父路径 $a,b,c$ 与 $e_3$ 才构成三角形。

<!-- bilingual-en:start -->
Misclassifying every reverse parent-edge reading would falsely mark every tree **with at least one edge** as cyclic. A one-vertex tree has no such reading and no cycle. A [[游走路径与环|closed walk]] going out and back on a single undirected edge repeats that edge; it is not a two-edge cycle.
<!-- bilingual-en:end -->

若把每次父边反读都判成环，就会把每棵**至少含一条边**的树误判有环。单点树没有这种反读，也没有环。沿同一条无向边走出再走回的[[游走路径与环|闭游走]]重复使用该边，并不是两边环。

<!-- bilingual-en:start -->
## Exclude the exact parent edge
<!-- bilingual-en:end -->

*排除精确的父边*

<!-- bilingual-en:start -->
While inspecting an adjacency entry for an edge $e$ from $u$ to $v$, first skip it if $e=\operatorname{parentEdge}[u]$. Otherwise, if $v$ is white, mark it, record $e$ as its parent edge and explore it immediately; after returning, continue with the next adjacency entry. If $v$ was gray at inspection, it lies on the active ancestor chain. Its parent path to $u$, followed by $e$, is a cycle. The path has distinct vertices, and $e$ is a different edge from all its tree edges. A black target gives no new active-chain certificate and can be ignored by this version of the detector.
<!-- bilingual-en:end -->

检查从 $u$ 经边 $e$ 到 $v$ 的邻接项时，若 $e=\operatorname{parentEdge}[u]$，先跳过它。否则，若 $v$ 为白色，就标记它、把 $e$ 记为其父边并立即探索；返回后继续下一个邻接项。若检查时 $v$ 为灰色，它就在活动祖先链上；从它沿父路径到达 $u$，再接 $e$，便是环。父路径顶点互异，而 $e$ 与其中每条树边都不同。这一检测版本可以忽略黑色目标，因为它不提供新的活动链证书。

<!-- bilingual-en:start -->
Why is this detector complete when restarted at every undiscovered root? Suppose a full DFS finished without a report although a cycle exists. For a cycle of length at least two, choose its first-finished vertex $u$. The cycle has two distinct edges incident to $u$, and at most one is its parent edge. Choose the other edge $e$ with neighbor $v$. When $u$ inspected $e$, $v$ could not have been black, because it would have finished earlier on the cycle; nor white, because its new recursive call would finish before $u$. Therefore it was gray, and $e$ should have produced a report. A self-loop instead reaches gray $u$ itself and is detected directly. This proves both detection and absence claims for a completed full search, including disconnected graphs.
<!-- bilingual-en:end -->

为什么从每个未发现根重启后，这个检测不会漏环？假设完整 DFS 结束而未报环，但图中有环。对长度至少为二的环，取其中最早完成的顶点 $u$。环上有两条不同的边关联于 $u$，至多一条是它的父边；取另一条 $e$，设其邻点为 $v$。检查 $e$ 时，$v$ 不可能为黑色，否则环上已有更早完成的点；也不可能为白色，否则新递归调用会先于 $u$ 完成。因此它应为灰色，$e$ 本应触发报告，矛盾。自环则直接返回灰色的 $u$，立即被检测。于是，完整搜索既能发现环，也能在遍历结束而无报告时证明无环；不连通图同样适用。

<!-- bilingual-en:start -->
## Edge identity and graph type
<!-- bilingual-en:end -->

## 图类型边界

<!-- bilingual-en:start -->
If two distinct parallel edges $e_1,e_2$ join $p$ and $u$, DFS may discover $u$ through $e_1$. From $u$, skip only $e_1$; $e_2$ reaches gray parent $p$ and closes the two-cycle using $e_1$ and $e_2$. Both edges have the same endpoints, but their identities differ. Thus “skip every neighbor equal to the parent vertex” is valid for this purpose in a simple graph, but is not a general multigraph rule. A program can store unique edge IDs, or first detect parallel edges and then use a simple-neighbor rule; these are different bookkeeping choices.
<!-- bilingual-en:end -->

若两条不同的平行边 $e_1,e_2$ 连接 $p,u$，DFS 可能经 $e_1$ 发现 $u$。在 $u$ 只跳过 $e_1$；$e_2$ 返回灰色父顶点 $p$，便与 $e_1$ 构成两边环。两条边的端点相同，身份却不同。因此，“跳过所有等于父顶点的邻居”在简单图的这一用途下可行，却不是通用的多重图规则。程序可以保存唯一边 ID，也可以先检测平行边，再使用简单邻居规则；两者是不同的记录方式。

<!-- bilingual-en:start -->
If self-loops are allowed, a loop at $u$ is not the edge that discovered $u$. It returns to gray $u$ and forms a one-edge cycle. In a simple graph, neither self-loops nor parallel edges exist, so cycles have length at least three. An empty graph or isolated vertex has no cycle. These length conventions concern the [[图类型的建模边界|chosen graph model]], not a choice of DFS order.
<!-- bilingual-en:end -->

若允许自环，$u$ 的自环不可能是发现 $u$ 所用的父边；它返回灰色的 $u$，构成一边环。简单图既无自环也无平行边，因此环至少长三。空图或孤立顶点均无环。这些长度约定来自[[图类型的建模边界|所选图模型]]，而不是 DFS 次序。

<!-- bilingual-en:start -->
Another valid bookkeeping rule examines each **underlying edge only once**, using an edge-seen flag to suppress its second incidence. At that edge's first examination, an undiscovered other endpoint makes it a tree edge. If the other endpoint is already discovered, the edge is not a previously selected tree edge, and the existing parent-tree path between its endpoints plus this new edge gives a cycle. Both endpoints belong to the same search tree: an undirected edge between different search runs would have discovered the later endpoint during the earlier run. “First examination” here refers to the edge identity across both endpoint lists, not merely the first time a particular list entry is read. Do not combine this rule with repeated incidence scans while forgetting which edges are already in the tree.
<!-- bilingual-en:end -->

另一种有效的记录规则是每条**底层边只检查一次**，用“边已检查”标记跳过它的另一端关联项。在该边首次检查时，若另一端未发现，就将它选为树边；若另一端已发现，它便不是先前已经选用的树边，已有父树中连接两端的路径加上这条新边即可成环。两端属于同一棵搜索树：若无向边连接不同搜索轮次，较早的一轮就会通过它发现较晚的端点。这里的“首次检查”指跨两端列表的边身份，而非某一个邻接项第一次被读。不能一面重复扫描关联项，一面又忘记哪些边已属于父树，再混用这一规则。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.006, Lecture 10](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf), PDF pp. 2–4: nested DFS, parent paths and active ancestors. The exact undirected-edge exception and earliest-finish completeness proof are developed explicitly above.
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=527|MIT Mathematics for Computer Science, §12.7]], PDF pp. 527–529: simple undirected walks and cycles, including why returning on the same edge is not a cycle. The loop and parallel-edge extensions follow the graph model stated in this card.
- [Sedgewick and Wayne, Algorithms, 4th ed., official Cycle.java](https://algs4.cs.princeton.edu/41graph/Cycle.java): an implementation that checks parallel edges before applying its parent-vertex exception. It illustrates the need to handle that boundary separately; it is not the edge-ID implementation described above.
<!-- bilingual-en:end -->

- MIT 6.006 Lecture 10，第 2–4 页：嵌套 DFS、父路径与活动祖先。正文另完整推导精确无向父边例外，以及最早完成顶点的完备性证明。
- MIT《Mathematics for Computer Science》§12.7，PDF 第 527–529 页：简单无向图的游走与环，包括同一边往返不成环。自环和平行边扩展遵循本卡明确的图模型。
- Sedgewick 与 Wayne《Algorithms》第四版官方 `Cycle.java`：先检测平行边，再应用按父顶点判断的例外。它说明这一边界需要单独处理，并非正文采用边 ID 的那一种实现。

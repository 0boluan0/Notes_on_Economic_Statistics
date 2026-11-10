---
aliases:
  - Directed DFS cycle certificate
  - 有向 DFS 回边与环
student_os: knowledge-atom
atom_id: MCS-GRAPH-017
atom_type: theorem
status: source-checked
part_of:
  - "[[图的基本结构、路径与遍历.canvas]]"
requires:
  - "[[DFS 可达性与非最短路径]]"
leads_to:
  - "[[无向 DFS 的父边例外]]"
---

<!-- bilingual-en:start -->
# Directed cycles in DFS
<!-- bilingual-en:end -->

*有向 DFS 的环证书*

<!-- bilingual-en:start -->
A search may meet a vertex it has seen before without having found a cycle. In a directed graph, the useful event is more specific: while exploring the current vertex, an edge leads back to a vertex whose recursive exploration is still active. The active parent chain then supplies the missing route that closes the cycle.
<!-- bilingual-en:end -->

搜索再次遇见旧顶点，并不一定找到了环。有向图中，真正有用的事件更具体：探索当前顶点时，一条边返回了递归探索仍在活动的顶点。此时，活动父链恰好提供闭合成环所需的另一段路线。

<!-- bilingual-en:start -->
Use standard [[DFS 可达性与非最短路径|DFS]] on a finite [[图类型的建模边界|directed graph]], either recursively or with equivalent stack frames that retain suspended scan positions. Give each vertex three possible states: **white** means undiscovered, **gray** means its call has begun but not finished, and **black** means its scan and all child calls have finished. At the moment an edge is inspected, gray vertices are precisely the active root-to-current-vertex chain, including the current vertex itself.
<!-- bilingual-en:end -->

在有限[[图类型的建模边界|有向图]]上使用标准[[DFS 可达性与非最短路径|DFS]]：可以递归，也可以用保存暂停扫描位置的等价栈帧。顶点分三种状态：**白色**未发现，**灰色**调用已开始但未完成，**黑色**自身扫描及全部子调用均已完成。在检查一条边的时刻，灰色顶点恰好组成从根到当前顶点的活动链，包括当前顶点自身。

> [!important] An edge back to the active chain · 返回活动链的边
> <!-- bilingual-en:start -->
> If DFS inspects $u\to v$ while $v$ is gray, the forward parent path from $v$ to $u$, followed by $u\to v$, is a [[游走路径与环|directed cycle]]. This edge is a **back edge**. Here the active-chain test includes $v=u$, so an allowed self-loop is a one-edge cycle.
> <!-- bilingual-en:end -->
>
> 若检查 $u\to v$ 时 $v$ 为灰色，从 $v$ 沿父树向前到达 $u$ 的路径，再接上 $u\to v$，就构成[[游走路径与环|有向环]]。这样的边叫**回边**。活动链判定包括 $v=u$，因此允许自环时，它本身就是长度为一的环。

<!-- bilingual-en:start -->
## Construct the certificate in the correct direction
<!-- bilingual-en:end -->

*按正确方向构造证书*

<!-- bilingual-en:start -->
Suppose DFS has active vertices $a,b,c$ with parent edges $a\to b$ and $b\to c$, and now inspects $c\to a$. The certificate is $a\to b\to c\to a$. To recover it from parent records, follow parents backward from $c$ until reaching $a$, reverse the resulting vertex list, then append the closing edge. The parent pointers are followed backward only to recover the list; the certified path traverses the tree edges forward.
<!-- bilingual-en:end -->

设当前活动顶点为 $a,b,c$，父边为 $a\to b$、$b\to c$，现在检查 $c\to a$。证书就是 $a\to b\to c\to a$。若从父记录恢复它，先从 $c$ 沿父指针倒着找到 $a$，把所得顶点列表反转，再接上闭合边。沿父指针倒走只是恢复列表的操作；真正证明的路径沿树边正向前进。

<!-- bilingual-en:start -->
Why is this a cycle rather than just a closed walk? The [[搜索标记与父森林|parent path]] has distinct vertices, since each vertex is discovered once. The closing edge returns to its first vertex without introducing any other repetition. If $u\ne v$, it cannot be one of those forward tree edges. If $u=v$, the parent segment has length zero and the self-loop alone is the certificate. Opposite edges $a\to b$ and $b\to a$ similarly give a two-edge directed cycle. Parallel edges with the same direction alone do not give a return route; record the chosen edge identities when they matter.
<!-- bilingual-en:end -->

为什么得到的是环，而不只是闭游走？[[搜索标记与父森林|父路径]]的顶点互不重复，因为每点只发现一次。闭合边返回起点，不会引入其他重复。若 $u\ne v$，它不可能是路径上的正向树边；若 $u=v$，父路径段长度为零，自环独自构成证书。相反方向的 $a\to b$、$b\to a$ 也构成两边有向环；只有同向平行边则没有返回路线。需要区分平行边时，应保留所选边的身份。

<!-- bilingual-en:start -->
A black neighbor does not provide this active-ancestor certificate. In the acyclic graph $a\to c$, $b\to c$, start from $a$: $c$ finishes, then $a$ finishes. A later search from $b$ sees black $c$. There is no parent path from $c$ to $b$, so this shared successor does not close a cycle. A single `visited` flag cannot distinguish this case from the gray case. Keep `active/on_stack` separately, or equivalent discovery-and-finish information; clear active status on normal return.
<!-- bilingual-en:end -->

黑色邻居不提供这种活动祖先证书。在无环图 $a\to c$、$b\to c$ 中，从 $a$ 开始，先完成 $c$，再完成 $a$；之后从 $b$ 搜索会看到黑色的 $c$。父记录中没有从 $c$ 到 $b$ 的路径，所以共享后继不会闭合成环。单个 `visited` 标记不能把这一情形与灰色情形区分开；还应记录 `active/on_stack`，或等价的发现/完成信息，并在正常返回时清除活动状态。

<!-- bilingual-en:start -->
## Why a full DFS finds a certificate whenever a cycle exists
<!-- bilingual-en:end -->

## 反向：有环时 full DFS 必遇到祖先边

<!-- bilingual-en:start -->
Run a [[全图搜索与强连通边界|full DFS]], restarting at each still-white vertex and retaining all old marks. Suppose it completed without finding a gray-target edge, but the graph contains a directed cycle. Choose the vertex $u$ on that cycle whose call finished first, and let $v$ be its next vertex on the cycle. The edge $u\to v$ was inspected before $u$ finished.
<!-- bilingual-en:end -->

进行[[全图搜索与强连通边界|完整 DFS]]，从仍为白色的顶点重启并保留旧标记。假设全部搜索完成后都没有遇到指向灰点的边，但图中存在有向环。取该环上**最早完成调用**的顶点 $u$，令 $v$ 是环上紧接着的顶点。$u$ 完成前必检查过 $u\to v$。

<!-- bilingual-en:start -->
At that inspection, $v$ cannot be white: DFS would immediately explore $v$ and finish its call before returning to finish $u$, contradicting the choice of $u$. It cannot be black either, because then a cycle vertex had already finished. Therefore $v$ was gray, giving the forbidden back edge. For a self-loop, $v=u$ is gray directly. This proves that a finite directed graph has no cycle **if and only if** a full DFS encounters no edge to its active chain. The proof does not require DFS to traverse a chosen cycle in consecutive cyclic order.
<!-- bilingual-en:end -->

检查时 $v$ 不可能为白色：否则 DFS 会立即探索它，并在返回完成 $u$ 之前完成 $v$，与 $u$ 最早完成矛盾。$v$ 也不可能为黑色，否则已有环上顶点更早完成。因此 $v$ 必为灰色，出现所排除的回边。自环时 $v=u$ 本来就是灰色。由此，有限有向图无环，**当且仅当**完整 DFS 没有遇到指向活动链的边。证明不要求 DFS 恰按所选环的次序连续走一圈。

<!-- bilingual-en:start -->
A single-source DFS can certify only the part [[图距离与可达性|reachable]] from that source. For example, an isolated source can finish while a separate component contains $a\to b\to a$. Finding no back edge in that one run does not prove the whole graph acyclic. Full DFS has no calls on the empty graph, which is acyclic. The active-state rule also presumes nested DFS: a generic LIFO worklist may contain pending siblings, not just ancestors. For undirected graphs, the [[无向 DFS 的父边例外|same parent edge read in reverse]] requires a different treatment.
<!-- bilingual-en:end -->

一次单源 DFS 只能判断该源点[[图距离与可达性|可达]]的部分。例如，孤立源点可以直接完成，而另一个分量中仍有 $a\to b\to a$；这一轮没遇到回边，不代表整图无环。空图的完整 DFS 无需调用，且空图无环。活动状态规则还依赖嵌套 DFS：普通 LIFO 工作表可能装着等待的兄弟节点，而非只有祖先。无向图中，[[无向 DFS 的父边例外|同一父边的反向读取]]需要另行处理。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.006, Lecture 10](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf), PDF pp. 2–4, especially p. 4: recursive DFS, full-root traversal, active ancestors and the cycle certificate. The earliest-finish argument above supplies an explicit equivalent converse; PDF p. 5 is the credits page.
- [Sedgewick and Wayne, Algorithms, 4th ed., official DirectedCycle.java](https://algs4.cs.princeton.edu/42digraph/DirectedCycle.java): distinct `marked` and `onStack` arrays, parent reconstruction, normal-return cleanup and iteration over undiscovered roots. The mathematical proof above establishes the certificate and includes the stated self-loop boundary.
<!-- bilingual-en:end -->

- MIT 6.006 Lecture 10，第 2–4 页，特别是第 4 页：递归 DFS、全根遍历、活动祖先与环证书。正文用最早完成顶点给出显式的等价逆向证明；PDF 第 5 页是版权页。
- Sedgewick 与 Wayne《Algorithms》第四版官方 `DirectedCycle.java`：分开保存 `marked` 和 `onStack`、沿父记录恢复环、正常返回时清理活动状态，以及遍历所有未发现根。正文的数学论证说明证书为何成立，并包含所述自环边界。

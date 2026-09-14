---
aliases:
  - "无向 DFS 返回父节点的边不是回边而其他已发现祖先才闭合成环"
  - Ignore the parent edge in undirected DFS cycle detection
  - 无向 DFS 的父边例外
student_os: knowledge-atom
atom_id: MCS-GRAPH-018
atom_type: diagnostic
status: source-checked
mastery_state: unassessed
part_of:
  - "[[图的基本结构、路径与遍历.canvas]]"
requires:
  - "[[有向 DFS 的环证书]]"
  - "[[图类型的建模边界]]"
leads_to:
  - "[[全图搜索与强连通边界]]"
---

# 无向 DFS 返回父节点的边不是回边而其他已发现祖先才闭合成环
<!-- bilingual-en:start -->
*In undirected DFS, the edge back to the parent is not a back edge; another discovered ancestor closes a cycle*
<!-- bilingual-en:end -->

> [!summary] 原子诊断
> 在 simple undirected graph 中，DFS 通过边 $\{p,u\}$ 从 parent $p$ 发现 $u$ 后，扫描 $u$ 的 adjacency list 必然再次看到 $p$。这只是同一条无向 tree edge 的反向读取，不是第二条边，也不形成 cycle。cycle detection 必须忽略“回到直接 parent 的那条 edge”；若 $u$ 另有边连向活动祖先 $a\ne p$，则 ancestor-to-$u$ tree path 加 $\{u,a\}$ 才构成 cycle。
> <!-- bilingual-en:start -->
> In a simple undirected graph, after DFS discovers $u$ from parent $p$ along edge $\{p,u\}$, scanning $Adj(u)$ necessarily sees $p$ again. This is the same undirected tree edge read from its other endpoint, not a second edge and not a cycle. Cycle detection must ignore that exact parent edge. An additional edge from $u$ to an active ancestor $a\ne p$ closes a cycle with the ancestor-to-$u$ tree path.
> <!-- bilingual-en:end -->

例如 tree $a-b-c$ 从 $a$ 开始 DFS。到 $b$ 时看到已发现的 $a$，到 $c$ 时看到已发现的 $b$；若把这两次都报成 back edges，任何非空 tree 都会被误判有环。加入边 $c-a$ 后，$c$ 看到非 parent 祖先 $a$，parent path $a-b-c$ 与 $c-a$ 才给出 triangle。
<!-- bilingual-en:start -->
For the tree $a-b-c$, DFS from $a$ sees discovered $a$ while processing $b$ and discovered $b$ while processing $c$. Treating either as a back edge would falsely label every nonempty tree cyclic. After adding $c-a$, vertex $c$ sees non-parent ancestor $a$, and parent path $a-b-c$ together with $c-a$ gives a triangle.
<!-- bilingual-en:end -->

## 图类型边界
<!-- bilingual-en:start -->
*Graph-type boundary*
<!-- bilingual-en:end -->

“忽略 parent”在 multigraph 中必须按**边 identity**而非只按 parent vertex 实现。若 $p,u$ 之间有两条 parallel edges，一条是 tree edge，另一条确实可与它组成长度 $2$ cycle；把所有 `neighbor == parent` 都跳过会漏报。simple graph 没有这种歧义。
<!-- bilingual-en:start -->
In a multigraph, “ignore the parent” must refer to the **identity of the parent edge**, not merely the parent vertex. If two parallel edges join $p$ and $u$, one may be the tree edge while the other closes a two-cycle. Skipping every `neighbor == parent` would miss it. A simple graph has no such ambiguity.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 无向 DFS 从 $u$ **第一次检查**边 $e=\{u,v\}$ 时，$v$ 已发现，而且 $e$ 不是发现 $u$ 时使用的那条 parent edge。为什么这时能给出 cycle certificate？
>
> **答案：**在 recursive DFS 首次检查 $e$ 时，若 $v$ 是 $u$ 的未发现 child，这条边会成为 tree edge，不会进入此分支。现在 $v$ 已发现且 $e$ 又不是那条 parent edge，因此 $e$ 是一条非树边；DFS tree 中已有一条连接 $u,v$ 的 parent path，加上 $e$ 就闭合成 cycle。关键是边是否属于 DFS tree，不是只看邻居当下是 gray 还是 black。
> <!-- bilingual-en:start -->
> When undirected DFS **first examines** an edge $e=\{u,v\}$ from $u$, vertex $v$ is already discovered and $e$ is not the parent edge that first discovered $u$. Why does this give a cycle certificate?
>
> **Answer:** When recursive DFS first examines $e$, an undiscovered $v$ would become a child and $e$ would become a tree edge, so this branch would not apply. Here $v$ is already discovered and $e$ is not the parent edge, making $e$ a non-tree edge. The parent tree already contains a path between $u$ and $v$; adding $e$ closes a cycle. The decisive fact is whether the edge belongs to the DFS tree, not merely whether the neighbor is gray or black at that moment.
> <!-- bilingual-en:end -->

## 来源与核验

- [MIT 6.006 Lecture 10](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf)，pp. 2、4：核验 DFS parent tree 与 ancestor-edge cycle certificate；无向 parent-edge 例外由同一无向 edge 在两个 adjacency lists 中各出现一次推出。
  <!-- bilingual-en:start -->
  Lecture 10 verifies DFS parent trees and ancestor-edge cycle certificates. The undirected parent-edge exception follows because the same undirected edge occurs in both endpoint adjacency lists.
  <!-- bilingual-en:end -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=527|MIT Mathematics for Computer Science, Section 12.7]]：核验 simple undirected graph 中沿同一边往返不是 cycle，而 cycle 至少长 $3$。
  <!-- bilingual-en:start -->
  Section 12.7 verifies that immediate reversal along the same simple undirected edge is not a cycle and that such cycles have length at least three.
  <!-- bilingual-en:end -->

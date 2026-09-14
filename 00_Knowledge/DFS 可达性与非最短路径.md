---
aliases:
  - "DFS 能找到单源可达区域但父路径通常不是最短路"
  - DFS gives reachability not shortest paths
  - DFS 单源可达与非最短父路径
student_os: knowledge-atom
atom_id: MCS-GRAPH-016
atom_type: procedure
status: source-checked
mastery_state: unassessed
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

# DFS 能找到单源可达区域但父路径通常不是最短路
<!-- bilingual-en:start -->
*DFS finds the single-source reachable region, but its parent paths are generally not shortest*
<!-- bilingual-en:end -->

> [!summary] 原子程序
> depth-first search（DFS）从源点 $s$ 出发，对当前顶点的某个未发现 outgoing neighbor 立即递归或压入 LIFO stack，沿一条分支尽可能深入；只有当前点再无未探索邻居时才回溯。它最终发现且只发现从 $s$ 可达的顶点，并给每个已发现顶点一条 parent path。
> <!-- bilingual-en:start -->
> Depth-first search (DFS) starts at source $s$ and immediately recurses into, or pushes onto a LIFO stack, an undiscovered outgoing neighbor. It follows one branch as deeply as possible and backtracks only when the current vertex has no unexplored neighbor. It discovers exactly the vertices reachable from $s$ and gives each one a parent path.
> <!-- bilingual-en:end -->

## 可达性正确性
<!-- bilingual-en:start -->
*Reachability correctness*
<!-- bilingual-en:end -->

DFS 不会离开源点可达区域，因为每次发现都沿一条真实边延长已有 parent path。反过来，若 $v$ 从 $s$ 可达，取一条 shortest path 上 $v$ 的前驱 $u$；按 path 长度归纳，DFS 会访问 $u$，而访问 $u$ 时必检查通向 $v$ 的边。若 $v$ 尚未发现就在此发现，否则它已经通过另一条路发现。故所有 reachable vertices 最终都被访问。
<!-- bilingual-en:start -->
DFS cannot leave the source-reachable region because each discovery extends an existing parent path along a real edge. Conversely, if $v$ is reachable, take its predecessor $u$ on a shortest path from $s$. Inducting on path length, DFS visits $u$ and therefore examines the edge to $v$. It discovers $v$ then if necessary, or $v$ was already discovered by another route. Thus every reachable vertex is eventually visited.
<!-- bilingual-en:end -->

## 为什么 parent path 不承担最短语义
<!-- bilingual-en:start -->
*Why the parent path has no shortest-path guarantee*
<!-- bilingual-en:end -->

DFS 的选择由邻接顺序和“先走到底”决定，不按距离层排序。图中即使有直达边 $s\to v$，DFS 也可能先沿 $s\to a\to b\to v$ 发现 $v$，从此不再改 parent。因此它提供 reachability certificate，却不计算 unweighted shortest distance。
<!-- bilingual-en:start -->
DFS choices follow adjacency order and depth-first commitment rather than distance layers. Even with a direct edge $s\to v$, DFS may first discover $v$ along $s\to a\to b\to v$ and never revise its parent. It therefore supplies a reachability certificate, not an unweighted shortest distance.
<!-- bilingual-en:end -->

DFS 的额外价值来自递归的嵌套结构：正在活动的祖先链、每个顶点的发现与完成时刻，以及 full DFS 的 finishing order。这些信息支持 cycle detection、topological sorting 等问题，而不是把 DFS 当成“另一种 BFS”。
<!-- bilingual-en:start -->
DFS gains additional power from its nested recursion structure: the active ancestor chain, discovery and finishing times, and the finishing order of a full DFS. These support cycle detection and topological sorting; DFS is not merely another implementation of BFS.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> DFS 返回 parent path $s-a-b-v$，图中另有边 $s-v$。DFS 的 reachability 结果是否错误？应怎样得到最短路？
>
> **答案：**reachability 仍正确，因为 parent path 是一条合法路径；错误只在把它称为 shortest。对 unweighted graph 从 $s$ 做 BFS，首次发现 $v$ 才给最少边数。
> <!-- bilingual-en:start -->
> DFS returns the parent path $s-a-b-v$, while the graph also contains edge $s-v$. Is its reachability result wrong, and how should the shortest path be found?
>
> **Answer:** Reachability is correct because the parent path is valid. Only the shortest-path claim would be wrong. Run BFS from $s$ in the unweighted graph to obtain minimum edge count at first discovery.
> <!-- bilingual-en:end -->

## 来源与核验

- [MIT 6.006 Lecture 10: Depth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf)，pp. 2–3：给出 DFS 伪代码、单源可达证明，并明确 parent tree 不必最短。
  <!-- bilingual-en:start -->
  Lecture 10 gives DFS pseudocode and the single-source reachability proof and explicitly states that its parent tree need not be shortest.
  <!-- bilingual-en:end -->

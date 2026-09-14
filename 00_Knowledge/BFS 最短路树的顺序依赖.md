---
aliases:
  - "BFS 父指针给出一棵受邻接顺序影响的无权最短路树"
  - BFS parent pointers form one shortest-path tree
  - BFS 最短路父树
student_os: knowledge-atom
atom_id: MCS-GRAPH-015
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[图的基本结构、路径与遍历.canvas]]"
requires:
  - "[[BFS 分层与无权最短路]]"
  - "[[搜索标记与父森林]]"
contrasts_with:
  - "[[DFS 可达性与非最短路径]]"
---

# BFS 父指针给出一棵受邻接顺序影响的无权最短路树
<!-- bilingual-en:start -->
*BFS parent pointers give one unweighted shortest-path tree selected by adjacency order*
<!-- bilingual-en:end -->

> [!summary] 原子区分
> BFS 首次从 $u$ 发现 $v$ 时记录 $parent[v]=u$。沿 parent pointers 从 $v$ 回到源点 $s$，恰走 $dist[v]$ 条边，因此得到一条 shortest path。所有 parent edges 合起来是一棵 shortest-path tree，但它通常只是多种合法结果中的一棵。
> <!-- bilingual-en:start -->
> When BFS first discovers $v$ from $u$, it records $parent[v]=u$. Following parents from $v$ back to source $s$ traverses exactly $dist[v]$ edges and therefore reconstructs a shortest path. The parent edges form a shortest-path tree, usually one of several valid possibilities.
> <!-- bilingual-en:end -->

若 $v$ 在同一层有两个可能前驱 $a,b$，先被扫描的那个会成为 parent。改变 adjacency-list 顺序可能把 $parent[v]$ 从 $a$ 换成 $b$，但不会改变 $dist[v]$。因此距离是图与源点决定的数学量；在 tie 存在时，parent tree 还受遍历顺序决定。
<!-- bilingual-en:start -->
If $v$ has two possible predecessors $a,b$ in the previous layer, whichever is scanned first becomes its parent. Changing adjacency-list order may replace $parent[v]=a$ by $parent[v]=b$ without changing $dist[v]$. Distance is determined by the graph and source; a parent tree may additionally depend on traversal order when ties exist.
<!-- bilingual-en:end -->

## 输出压缩与边界
<!-- bilingual-en:start -->
*Output compression and boundary*
<!-- bilingual-en:end -->

显式列出从 $s$ 到每个顶点的整条 path 可能需要 $\Omega(|V|^2)$ 总空间；parent array 只占 $O(|V|)$，需要某个终点时再反向重建。它不会列出**所有** shortest paths：若要计数或枚举全部最短路，必须额外保留同层的所有合法 predecessors。
<!-- bilingual-en:start -->
Explicitly storing a complete source path for every vertex may require $\Omega(|V|^2)$ total space. A parent array uses only $O(|V|)$, reconstructing one path backward when needed. It does not list **all** shortest paths; counting or enumerating all of them requires retaining every predecessor that achieves the shortest distance.
<!-- bilingual-en:end -->

BFS tree 也能作为 reachability certificate：某点有 parent chain 就说明它可达；没有被发现只在搜索确实扫描完所有 reachable adjacency entries 后才证明不可达。对 weighted graph，这棵树仍是最少边数树，不自动成为 minimum-total-weight tree。
<!-- bilingual-en:start -->
The BFS tree is also a reachability certificate: a parent chain proves that a vertex is reachable. Absence from the tree proves unreachability only after all reachable adjacency entries have been processed. In a weighted graph the tree still minimizes edge count, not necessarily total weight.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 两次 BFS 从同一源点得到不同 parent array，但每个顶点的 distance 完全相同。这是否说明其中一次错误？
>
> **答案：**不一定。若存在多条等长 shortest paths，邻接顺序不同可选择不同前驱；只要每个非根顶点满足 $dist[v]=dist[parent[v]]+1$ 且 parent chain 合法到达源点，两棵树都可能正确。
> <!-- bilingual-en:start -->
> Two BFS runs from the same source return different parent arrays but identical distances. Must one be wrong?
>
> **Answer:** Not necessarily. Multiple equal-length shortest paths let different adjacency orders select different predecessors. Both trees may be correct if every non-root vertex satisfies $dist[v]=dist[parent[v]]+1$ and its parent chain follows valid edges to the source.
> <!-- bilingual-en:end -->

## 来源与核验

- [MIT 6.006 Lecture 9](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/196a95604877d326c6586e60477b59d4_MIT6_006S20_lec9.pdf)，pp. 3–4：定义 shortest-path tree 的 parent 压缩表示，并在首次分层发现时写入 parent。
  <!-- bilingual-en:start -->
  Lecture 9 defines the parent representation of a shortest-path tree and assigns a parent at first layered discovery.
  <!-- bilingual-en:end -->

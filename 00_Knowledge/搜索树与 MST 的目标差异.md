---
aliases:
  - "搜索树给可达或最短路证书而 MST 最小化全树总权重"
  - "Search trees and minimum spanning trees solve different objectives"
  - "搜索树与 MST 的区别"
student_os: knowledge-atom
atom_id: MCS-GRAPH-029
atom_set: acyclic-graphs-trees-dags
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[搜索标记与父森林]]"
  - "[[BFS 最短路树的顺序依赖]]"
  - "[[DFS 可达性与非最短路径]]"
  - "[[生成树的存在条件]]"
  - "[[最小生成树的定义与唯一性]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

# 搜索树给可达或最短路证书而 MST 最小化全树总权重
<!-- bilingual-en:start -->
*A search tree certifies reachability or shortest paths, while an MST minimizes total tree weight*
<!-- bilingual-en:end -->

> [!summary] 原子区分
> BFS/DFS 的 parent edges 形成以搜索根为中心的 search tree：DFS 只保证给出可达 path，BFS 在 unweighted graph 中还保证每条 root-to-vertex parent path 是 shortest。更一般的 shortest-path tree 也优化固定 root 到各点的 distances。MST 没有指定根；它最小化整棵 spanning tree 的 edge-weight 总和。根到点的路径目标与全树总权重目标没有一般蕴含关系。
> <!-- bilingual-en:start -->
> BFS and DFS parent edges form a rooted search tree. DFS certifies reachability; in an unweighted graph, BFS also makes each root-to-vertex parent path shortest. More generally, a shortest-path tree optimizes distances from a fixed root. An MST has no distinguished root and minimizes the total edge weight of the whole spanning tree. Neither objective generally implies the other.
> <!-- bilingual-en:end -->

例如 triangle 有 $w(sa)=100,w(sb)=1,w(ba)=1$。从 $s$ 做 BFS 会用一跳边 $sa$ 与 $sb$ 首次发现两个点，所得 tree 总权重 $101$；MST 选 $sb,ba$，总权重 $2$。BFS 没有出错：它最小化的是从 $s$ 出发的**边数**，不是全树权重。
<!-- bilingual-en:start -->
In a triangle with $w(sa)=100,w(sb)=1,w(ba)=1$, BFS from $s$ may choose the one-edge discoveries $sa$ and $sb$, producing total weight $101$. The MST uses $sb$ and $ba$ for total weight $2$. BFS is correct for edge-count distance; it was never optimizing total tree weight.
<!-- bilingual-en:end -->

即使明确求 weighted shortest-path tree，也不等于 MST。若 triangle 的权重为 $w(sa)=2,w(sb)=2,w(ab)=1$，从 $s$ 到 $a,b$ 的 shortest distances 都是 $2$，所以两条直连边组成一棵 shortest-path tree，总权重 $4$；MST 则取 $ab$ 和任一条权重 $2$ 的边，总权重 $3$。
<!-- bilingual-en:start -->
Even a weighted shortest-path tree need not be an MST. In a triangle with $w(sa)=2,w(sb)=2,w(ab)=1$, both shortest distances from $s$ are $2$, so the two direct edges form a shortest-path tree of total weight $4$. An MST instead takes $ab$ and either weight-$2$ edge, for total weight $3$.
<!-- bilingual-en:end -->

若所有边权相同，任意 spanning tree 都有 $n-1$ 条边并因而同权；此时一棵覆盖全图的 BFS/DFS tree 也是某个 MST，但原因是目标退化为所有 spanning trees 并列，而不是搜索算法自动解决了一般 MST。
<!-- bilingual-en:start -->
If all edge weights are equal, every spanning tree has $n-1$ equal-weight edges, so any spanning BFS/DFS tree is also an MST. This happens because all spanning trees tie, not because graph search solves the general MST problem.
<!-- bilingual-en:end -->

> [!question]- 自检
> 一棵 BFS tree 中每个根到点的 path 都最短，为什么仍可能不是 MST？
>
> **答案：**这些是多个以同一根为起点的边数最短条件；MST 比较的是整组被选边的权重总和，两种目标没有一般蕴含关系。

## 来源与核验

- [MIT 6.006 Lecture 9, Breadth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/196a95604877d326c6586e60477b59d4_MIT6_006S20_lec9.pdf)：核对 shortest-path tree 是固定 source 到各点的路径证书，以及 BFS 的 unweighted shortest-path 语义。
- [MIT 6.006 Lecture 10, Depth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf)：核对 DFS parent tree 只承担 reachability 而非 shortest-path 语义。
- [MIT 6.006 Lecture 11, Weighted Shortest Paths](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/aa57a9785adf925bc85c1920f53755a0_MIT6_006S20_lec11.pdf)：核对 weighted distance 与 single-source shortest-path 目标；它仍不同于 spanning-tree 总权重目标。
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=539|MIT Mathematics for Computer Science, §12.11.4]]：核对 MST 的全树总权重目标。

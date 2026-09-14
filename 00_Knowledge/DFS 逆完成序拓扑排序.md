---
aliases:
  - "DAG 的 full DFS 逆完成顺序是一种 topological order"
  - "Reverse DFS finishing order topologically sorts a DAG"
  - "DFS 逆完成顺序拓扑排序"
student_os: knowledge-atom
atom_id: MCS-GRAPH-035
atom_set: acyclic-graphs-trees-dags
atom_type: algorithm
status: source-checked
mastery_state: unassessed
requires:
  - "[[DAG 与拓扑序等价]]"
  - "[[有向 DFS 的环证书]]"
  - "[[全图搜索与强连通边界]]"
related:
  - "[[Kahn 拓扑排序与环证书]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

# DAG 的 full DFS 逆完成顺序是一种 topological order
<!-- bilingual-en:start -->
*Reverse full-DFS finishing order is a topological order of a DAG*
<!-- bilingual-en:end -->

> [!summary] 原子算法
> 对 finite DAG 运行 full DFS，记录每个顶点完成递归处理的 finishing order；把这个顺序反转，就得到一种 topological order。必须是覆盖所有顶点的 **full** DFS，而不是只从一个 root 搜索其 reachable region。
> <!-- bilingual-en:start -->
> Run full DFS on a finite DAG and record the order in which recursive visits finish. Reversing that finishing order gives a topological order. The search must be **full**, covering every vertex rather than only one root's reachable region.
> <!-- bilingual-en:end -->

对任意 edge $u\to v$，证明 $v$ 必先于 $u$ 完成。若 DFS 先访问 $u$ 而 $v$ 尚未发现，处理 $u$ 的可达后继会在 $u$ 完成前完成 $v$；若 $v$ 已发现，DAG 排除了 $v$ 仍为 $u$ 活动祖先的情况，而已完成的 $v$ 当然更早结束。若 DFS 在另一个 tree 中先完成 $v$，结论也直接成立。因此逆序把 $u$ 放在 $v$ 前。
<!-- bilingual-en:start -->
For every edge $u\to v$, vertex $v$ finishes before $u$. If $u$ is visited first while $v$ is undiscovered, DFS completes the reachable visit to $v$ before finishing $u$. If $v$ was already seen, acyclicity rules out its being an active ancestor of $u$; a finished $v$ already ended earlier. The same is immediate when $v$ belongs to an earlier DFS tree. Reversal therefore puts $u$ before $v$.
<!-- bilingual-en:end -->

若 full DFS 发现指向活动祖先的 edge，应使用已有的 directed-cycle certificate 并拒绝输出 topological order。对任意 digraph 机械反转 finishing order 会产生一个排列，但在 cyclic graph 上它不满足所有 edge constraints。
<!-- bilingual-en:start -->
If full DFS finds an edge to an active ancestor, use the directed-cycle certificate and reject topological sorting. Reversing finishing times always produces a permutation, but on a cyclic graph that permutation violates at least one edge constraint.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么只从 source $s$ 做一次 DFS 后反转完成顺序，可能漏掉合法 DAG 的顶点？
>
> **答案：**DAG 可以不连通，也可以有不从 $s$ 可达的另一个 source/component；单源 DFS 根本没有访问这些顶点，所得列表不是全顶点排列。

## 来源与核验

- [MIT 6.006 Lecture 10, Depth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf)：逐边证明 DAG 的 reverse finishing order 是 topological order，并把 active-ancestor edge 用作 cycle certificate。

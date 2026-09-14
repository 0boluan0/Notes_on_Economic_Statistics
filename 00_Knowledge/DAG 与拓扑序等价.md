---
aliases:
  - "有限 digraph 是 DAG 当且仅当它存在 topological order"
  - "Directed Acyclic Graphs"
  - "DAG"
  - "Topological Sort"
  - "DAG 与拓扑排序"
student_os: knowledge-atom
atom_id: MCS-GRAPH-033
atom_set: acyclic-graphs-trees-dags
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Tree 与 DAG 的区别]]"
  - "[[度数、入度与出度]]"
  - "[[游走路径与环]]"
related:
  - "[[有向 DFS 的环证书]]"
leads_to:
  - "[[Kahn 拓扑排序与环证书]]"
  - "[[DFS 逆完成序拓扑排序]]"
  - "[[拓扑序的线性扩张]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

# 有限 digraph 是 DAG 当且仅当它存在 topological order
<!-- bilingual-en:start -->
*A finite digraph is a DAG exactly when it has a topological order*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> topological order 是顶点的一种线性排列，使每条 edge $u\to v$ 都把 $u$ 放在 $v$ 之前。对 finite digraph，以下条件等价：
> 1. 没有 directed cycle，即它是 DAG；
> 2. 存在 topological order；
> 3. 每个非空 induced subgraph 至少有一个 indegree-$0$ source；
> 4. 每个非空 induced subgraph 至少有一个 outdegree-$0$ sink。
> <!-- bilingual-en:start -->
> A topological order is a linear ordering of the vertices in which every edge $u\to v$ puts $u$ before $v$. For a finite digraph, the following are equivalent: it is a DAG; it has a topological order; every nonempty induced subgraph has an in-degree-zero source; and every nonempty induced subgraph has an out-degree-zero sink.
> <!-- bilingual-en:end -->

若有限 digraph 没有 source，则从任一点不断选一条 incoming edge 逆行；有限性迫使某个顶点重复，重复段给出 directed cycle。因此 finite DAG 有 source，删除它后仍是 DAG，递归输出 sources 就构造出 topological order。反向，若存在 directed cycle，沿 cycle 每条 edge 都要求排列位置严格增加，最终却回到起点，不可能成立。
<!-- bilingual-en:start -->
If a finite digraph has no source, repeatedly follow an incoming edge backwards. Finiteness forces a repeated vertex and hence a directed cycle. A finite DAG therefore has a source; deleting it preserves acyclicity, so recursively outputting sources constructs a topological order. Conversely, positions cannot increase strictly around a directed cycle and return to the starting vertex.
<!-- bilingual-en:end -->

这里的 indegree/outdegree 必须在**当前 induced subgraph** 内重新计算。原图中有 incoming edge 的顶点，可能在其所有前驱被删除后成为新 source；这正是递归构造能够继续的原因。
<!-- bilingual-en:start -->
In-degree and out-degree are recomputed in the **current induced subgraph**. A vertex with incoming edges in the original graph may become a source after all its predecessors are removed.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么“有一个 topological order”足以证明图没有 directed cycle？
>
> **答案：**cycle 上每一步都使顶点位置严格增大，但最后一条边又要求起点排在自身之后，形成矛盾。

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=441|MIT Mathematics for Computer Science, Definitions 10.5.1–10.5.3 and Theorem 10.5.4]]：核对 DAG、topological sort、minimal/source 递归构造与有限性边界。
- [MIT 6.006 Lecture 10, Depth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf)：核对 DAG 当且仅当存在 topological order 的算法课程表述。

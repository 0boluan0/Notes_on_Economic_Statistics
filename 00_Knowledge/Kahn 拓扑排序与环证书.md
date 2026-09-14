---
aliases:
  - "Kahn 算法要么输出 topological order 要么留下含环子图"
  - "Kahn algorithm and cycle certificate"
  - "Kahn 拓扑排序"
student_os: knowledge-atom
atom_id: MCS-GRAPH-034
atom_set: acyclic-graphs-trees-dags
atom_type: algorithm
status: source-checked
mastery_state: unassessed
requires:
  - "[[DAG 与拓扑序等价]]"
  - "[[度数、入度与出度]]"
related:
  - "[[有向 DFS 的环证书]]"
  - "[[拓扑序的线性扩张]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

# Kahn 算法要么输出 topological order 要么留下含环子图
<!-- bilingual-en:start -->
*Kahn's algorithm either outputs a topological order or leaves a cyclic subgraph*
<!-- bilingual-en:end -->

> [!summary] 原子算法
> Kahn algorithm 维护当前 indegree-$0$ sources。反复任选一个 source 输出，删除它的 outgoing edges，并把新变成 source 的顶点加入候选集。若输出全部顶点，所得序列是 topological order；若候选集提前为空而仍有顶点，剩余 induced subgraph 必含 directed cycle。
> <!-- bilingual-en:start -->
> Kahn's algorithm maintains the current in-degree-zero sources. It repeatedly outputs any source, deletes its outgoing edges, and adds newly created sources. If every vertex is output, the sequence is topological. If the source set becomes empty while vertices remain, the remaining induced subgraph contains a directed cycle.
> <!-- bilingual-en:end -->

正确性来自删除时刻：一个顶点只有在所有 incoming edges 都已随其前驱删除后才输出，所以每条 $u\to v$ 都先输出 $u$。失败时，剩余每个顶点至少有一条来自剩余集合的 incoming edge；在有限集合内不断沿入边逆行必重复顶点，显式记录这条 predecessor chain 就能提取 directed cycle。
<!-- bilingual-en:start -->
When a vertex is output, all its incoming edges have already disappeared with earlier predecessors, so every edge points forward in the output. On failure, every remaining vertex has an incoming edge from the remainder. Following such predecessors in a finite set repeats a vertex; recording the chain extracts a directed cycle.
<!-- bilingual-en:end -->

“算法停住”本身给出的是可核查的 residual-subgraph 证书；若接口要求返回具体 cycle，还需执行上述 predecessor-following extraction，不能把“没有 source”误写成已经列出了 cycle。用 adjacency lists、初始 indegrees 与 queue/set 实现时，总工作为 $O(|V|+|E|)$。
<!-- bilingual-en:start -->
Stopping supplies a checkable residual-subgraph certificate. Returning an explicit cycle additionally requires the predecessor-following extraction. With adjacency lists, stored in-degrees, and a queue or set, total work is $O(|V|+|E|)$.
<!-- bilingual-en:end -->

> [!question]- 自检
> Kahn 只输出了 8 个顶点便没有 source，而图还有 3 个顶点。能否把前 8 个顶点称为整个图的 topological order？
>
> **答案：**不能。topological order 必须包含全部顶点；剩余三点构成的 induced subgraph 含 directed cycle，证明原图不是 DAG。

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=442|MIT Mathematics for Computer Science, §10.5.1]]：核对反复选 minimal/source 顶点的 topological-sort 构造。
- [[01_Math/07-Mathematics for Computer Science/02_Structures.md#17.1 DAG 与拓扑排序|Session 17.1]]：核对课程的 indegree-$0$ 实现与 remaining-subgraph 反证。

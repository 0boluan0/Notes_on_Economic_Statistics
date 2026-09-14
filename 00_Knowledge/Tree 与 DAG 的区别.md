---
aliases:
  - "无向 tree 与 DAG 共享无环但不是同一种结构"
  - "Trees and DAGs are different acyclic structures"
  - "Tree 与 DAG 的区别"
student_os: knowledge-atom
atom_id: MCS-GRAPH-023
atom_set: acyclic-graphs-trees-dags
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[图类型的建模边界]]"
  - "[[游走路径与环]]"
leads_to:
  - "[[有限树的等价刻画]]"
  - "[[DAG 与拓扑序等价]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

# 无向 tree 与 DAG 共享无环但不是同一种结构
<!-- bilingual-en:start -->
*Undirected trees and DAGs are different structures even though both are acyclic*
<!-- bilingual-en:end -->

> [!summary] 原子区分
> **tree** 是 connected、acyclic 的无向 simple graph；**directed acyclic graph（DAG）** 是没有 directed cycle 的 digraph。tree 必须连通，DAG 可以不连通；tree 排除无向 cycle，DAG 只排除沿箭头闭合的 cycle。
> <!-- bilingual-en:start -->
> A **tree** is a connected, acyclic, undirected simple graph. A **directed acyclic graph (DAG)** is a digraph with no directed cycle. A tree must be connected, while a DAG need not be; a tree excludes undirected cycles, while a DAG excludes only cycles that follow edge directions.
> <!-- bilingual-en:end -->

给无向 tree 的每条边任意选方向，一定得到 DAG：若出现 directed cycle，忽略方向后就会在原 tree 中出现无向 cycle。反向不成立。比如
$$
A\to B,\quad A\to C,\quad B\to D,\quad C\to D
$$
没有 directed cycle，但忽略方向后得到 $A-B-D-C-A$；因此这个 DAG 的 underlying undirected graph 不是 tree，$A$ 到 $D$ 也有两条 directed paths。
<!-- bilingual-en:start -->
Arbitrarily orienting every edge of an undirected tree always produces a DAG: any directed cycle would become an undirected cycle after directions were forgotten. The converse fails. The diamond $A\to B,A\to C,B\to D,C\to D$ is a DAG, yet its underlying undirected graph contains the cycle $A-B-D-C-A$, and it has two directed paths from $A$ to $D$.
<!-- bilingual-en:end -->

所以“无环”必须连同图类型一起读。unique path、删边即断开和 $n-1$ 条边是 tree 分支的结论；source、topological order 和依赖调度是 DAG 分支的结论，不能跨分支直接套用。
<!-- bilingual-en:start -->
“Acyclic” must therefore be read together with the graph type. Unique paths, disconnection after deleting an edge, and the $n-1$ edge count belong to the tree branch. Sources, topological orders, and precedence scheduling belong to the DAG branch.
<!-- bilingual-en:end -->

> [!question]- 自检
> “图没有 directed cycle”能否推出任意两点间至多有一条 directed path？
>
> **答案：**不能。上面的 diamond 同时有 $A\to B\to D$ 与 $A\to C\to D$，却没有回到上游的 directed path，因此仍是 DAG。

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=439|MIT Mathematics for Computer Science, §10.5]]：核对 DAG 只排除 directed cycles。
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=535|MIT Mathematics for Computer Science, §12.11]]：核对 forest/tree 是无向 simple graph 分支中的 acyclic/connected-acyclic 结构。
- [[01_Math/07-Mathematics for Computer Science/02_Structures.md#Session 17 — Directed Acyclic Graphs|Session 17]] 与 [[01_Math/07-Mathematics for Computer Science/02_Structures.md#Session 21 — Trees and Minimum Spanning Trees|Session 21]]：核对课程的有向、无向术语边界。

---
aliases:
  - "有限 forest 若有 c 个连通分量就有 n-c 条边"
  - "A finite forest with c components has n minus c edges"
  - "Forest 的 n-c 边公式"
student_os: knowledge-atom
atom_id: MCS-GRAPH-025
atom_set: acyclic-graphs-trees-dags
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[有限树的等价刻画]]"
  - "[[无向图连通分量]]"
related:
  - "[[生成树的存在条件]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

# 有限 forest 若有 c 个连通分量就有 n-c 条边
<!-- bilingual-en:start -->
*A finite forest with $c$ connected components has $n-c$ edges*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> forest 是 acyclic 的无向 simple graph，每个 connected component 都是 tree。若有限 forest 有 $n$ 个顶点、$c$ 个 components，则
> $$|E|=n-c.$$
> 特别地，有限 tree 有 $c=1$，所以恰有 $n-1$ 条边。
> <!-- bilingual-en:start -->
> A forest is an acyclic undirected simple graph, and every connected component is a tree. If a finite forest has $n$ vertices and $c$ components, then $|E|=n-c$. A finite tree has $c=1$ and therefore exactly $n-1$ edges.
> <!-- bilingual-en:end -->

也可以不先借用 tree 的 $n_i-1$ 公式，直接数 component 的变化。起初只有 $n$ 个 isolated vertices，共 $n$ 个 components；逐条加入 forest 的边时，每条边的两端必须位于不同的当前 components，否则已有 path 加上该边就会闭合 cycle。因此每加入一边，component 数恰减 $1$。若最终加入 $|E|$ 条边后剩 $c$ 个 components，就有
$$
c=n-|E|,
$$
也就是 $|E|=n-c$。isolated vertex 始终作为一个零边 component 被正确计入。
<!-- bilingual-en:start -->
The formula can be proved directly without first importing the $n_i-1$ count for each component. Start with $n$ isolated vertices and hence $n$ components. Add the forest edges one at a time. Every new edge must join two distinct current components; otherwise an existing path between its endpoints together with the new edge would create a cycle. Each edge therefore reduces the component count by exactly one. After all $|E|$ edges have been added, $c=n-|E|$, or $|E|=n-c$. Isolated vertices remain correctly counted as zero-edge components.
<!-- bilingual-en:end -->

“$n$ 个顶点、$n-1$ 条边”**单独**不能证明 tree。triangle 加一个 isolated vertex 有 $n=4$、$|E|=3$，却同时 disconnected 且 cyclic。要用边数刻画 tree，还必须再知道 connected 或 acyclic：connected+$n-1$ 排除多余环边；acyclic+$n-1$ 由 $n-c=n-1$ 推出 $c=1$。
<!-- bilingual-en:start -->
The count $n-1$ alone does not prove that a graph is a tree. A triangle plus an isolated vertex has four vertices and three edges but is disconnected and cyclic. Combine the count with connectivity or acyclicity. In the latter case, $n-c=n-1$ forces $c=1$.
<!-- bilingual-en:end -->

> [!question]- 自检
> 一个有 20 个顶点、4 个 components 的 forest 有多少条边？如果只知道某图有 16 条边，能否反推它是 forest？
>
> **答案：**forest 有 $20-4=16$ 条边；单知边数不能反推无环，还要知道 component 数和相应结构条件。

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=537|MIT Mathematics for Computer Science, Theorems 12.11.5–12.11.6]]：核对 component 数、顶点数和边数的精确关系，以及 tree 的有限边数刻画。
- [[01_Math/07-Mathematics for Computer Science/02_Structures.md#21.2 forest、leaves 与二着色|Session 21.2]]：核对本课程的 $n-c$ 记号与 isolated-component 边界。

---
aliases:
  - "Prim 反复选择当前树与外部之间的最轻边"
  - "Prim algorithm for minimum spanning trees"
  - "Prim 算法"
student_os: knowledge-atom
atom_id: MCS-GRAPH-032
atom_set: acyclic-graphs-trees-dags
atom_type: algorithm
status: source-checked
mastery_state: unassessed
requires:
  - "[[MST 割性质]]"
related:
  - "[[Kruskal 算法]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

# Prim 反复选择当前树与外部之间的最轻边
<!-- bilingual-en:start -->
*Prim repeatedly selects the lightest edge from the current tree to the outside*
<!-- bilingual-en:end -->

> [!summary] 原子算法
> Prim 从任意 root $r$ 和顶点集 $S=\{r\}$ 开始。每一步选择 cut $(S,V\setminus S)$ 上的 minimum-weight crossing edge $(u,v)$，其中 $u\in S,v\notin S$，再把 $v$ 与该边加入。finite connected graph 上，$n-1$ 步后得到 MST。
> <!-- bilingual-en:start -->
> Prim starts from any root $r$ with $S=\{r\}$. At each step it chooses a minimum-weight edge $(u,v)$ crossing $(S,V\setminus S)$, with $u\in S$ and $v\notin S$, and adds $v$ and that edge. On a finite connected graph, the result after $n-1$ additions is an MST.
> <!-- bilingual-en:end -->

当前已选边全部位于 $S$ 内，所以 cut respects 它们；选中的 crossing edge 由 cut property 保证 safe。新边只把一个外部顶点接到现有 tree，不可能成环。connected 保证在 $S\ne V$ 时总有 crossing edge；若找不到，单次 Prim 只完成了 root 所在 connected component 的 MST，并没有覆盖其余顶点。要在 disconnected graph 上得到 minimum spanning forest，必须从每个尚未覆盖的 component 重新启动；不能把一次提前停止称为全图 MST。
<!-- bilingual-en:start -->
All previously selected edges lie inside $S$, so the cut respects them and the light crossing edge is safe. Adding one outside vertex by one crossing edge cannot create a cycle. Connectivity guarantees a crossing edge whenever $S\ne V$. If none exists, one run of Prim has completed only the MST of the root's connected component. Obtaining a minimum spanning forest on a disconnected graph requires restarting from every uncovered component; an early stop is not an MST of the whole graph.
<!-- bilingual-en:end -->

Kruskal 同时维护多个 forest components，并按全局 edge order 合并；Prim 始终只生长一棵 rooted tree。二者都用 cut property，但当前 cut 的来源不同。ties 可让不同 root 或 tie-breaking 产生不同 MST。
<!-- bilingual-en:start -->
Kruskal maintains several forest components and merges them in global edge order; Prim grows one rooted tree throughout. Both use the cut property, but their cuts come from different states. Ties can make the root or tie-breaking select different MSTs.
<!-- bilingual-en:end -->

> [!question]- 自检
> Prim 当前 $S$ 内有一条全图最轻边，还需要再次选择它吗？
>
> **答案：**不需要。两端都在 $S$ 内的边不跨当前 cut；加入它只会在已有 tree 内闭环。Prim 比较的是恰有一端在 $S$ 的 edges。

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_SpaingTrees.pdf|MIT 6.042J Spanning Trees slides]]：核对“从任意顶点持续生长一棵 tree”的 Prim 课程接口。
- [MIT 6.046J Lecture 4, Minimum Spanning Trees II](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2012/bb54568c86c60d86b25e9c1111470384_MIT6_046JS12_lec04.pdf)：核对 Prim 的 cut、safe-edge 不变量与 rooted-tree 增长步骤。

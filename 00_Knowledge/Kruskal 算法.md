---
aliases:
  - "Kruskal 按权重扫描并只合并不同 forest components"
  - "Kruskal algorithm for minimum spanning trees"
  - "Kruskal 算法"
student_os: knowledge-atom
atom_id: MCS-GRAPH-031
atom_set: acyclic-graphs-trees-dags
atom_type: algorithm
status: source-checked
mastery_state: unassessed
requires:
  - "[[Forest 的边数公式]]"
  - "[[MST 割性质]]"
related:
  - "[[无向 DFS 的父边例外]]"
  - "[[Prim 算法]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

# Kruskal 按权重扫描并只合并不同 forest components
<!-- bilingual-en:start -->
*Kruskal scans edges by weight and joins only distinct forest components*
<!-- bilingual-en:end -->

> [!summary] 原子算法
> Kruskal 从全体顶点、空边集 $F$ 开始，按 nondecreasing weight 扫描 edges。若一条边的两端位于 $F$ 的不同 components，就接受它；若两端已在同一 component，加入它会成环，因此跳过。connected graph 中，接受 $n-1$ 条边后得到 MST。
> <!-- bilingual-en:start -->
> Kruskal starts with all vertices and no selected edges. It scans edges in nondecreasing weight order, accepting an edge exactly when its endpoints lie in different current components. An edge within one component would create a cycle and is skipped. On a connected graph, the $n-1$ accepted edges form an MST.
> <!-- bilingual-en:end -->

在上述 connected setting 中，不变量一是 $F$ 始终为 forest；不变量二是 $F$ 始终可扩展为某个 MST。对将被接受的 $e=(u,v)$，取 $u$ 所在当前 component $S$。当前 $F$ 没有边跨 cut $(S,V\setminus S)$，所以 cut respects $F$。若存在更轻的 crossing edge，它较早被扫描时两端也不可能已连通——components 只会合并，而它们此刻仍位于 cut 两侧——所以当时必会被接受，反而会使它跨越当前 component，矛盾。因此 $e$ 是该 cut 的 light edge，cut property 保证它 safe。
<!-- bilingual-en:start -->
In the connected setting above, the selected set remains a forest and remains extendable to an MST. For the next accepted edge $e=(u,v)$, let $S$ be the current component of $u$. The cut $(S,V\setminus S)$ respects the forest. If a strictly lighter crossing edge existed, its endpoints could not have been connected when it was processed: components only merge, yet those endpoints still lie on opposite sides of the current component cut. It would therefore have been accepted, contradicting the definition of the current component. Thus $e$ is light across this cut and is safe by the cut property.
<!-- bilingual-en:end -->

edge-weight ties 只改变同一权重层的处理次序，可能选出不同 MST，不破坏正确性。component 检查通常由 disjoint-set/union–find 实现；若输入 disconnected，算法最终得到的是每个 component 上的 minimum spanning forest，而不是一棵 spanning tree。
<!-- bilingual-en:start -->
Ties may change the order within one weight class and produce different MSTs without invalidating correctness. Disjoint-set union normally implements the component test. On a disconnected input, the result is a minimum spanning forest, not one spanning tree.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么 Kruskal 不能“按权重从小到大全部接受”，而必须检查 endpoints 是否已连通？
>
> **答案：**同一 component 内两点已有 path；再加它们之间的边会闭合 cycle，使结果不再是 tree，也浪费一条不能改善连通性的边。

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_SpaingTrees.pdf|MIT 6.042J Spanning Trees slides]]：核对从 spanning forest 合并 components 的课程算法视角。
- [MIT 6.046J Lecture 3, Minimum Spanning Trees I](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2012/resources/mit6_046js12_lec03/) 与 [Lecture 4](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2012/bb54568c86c60d86b25e9c1111470384_MIT6_046JS12_lec04.pdf)：核对 Kruskal 的 forest invariant、safe-edge 证明和 component 数据结构。

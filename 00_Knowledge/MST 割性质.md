---
aliases:
  - "MST cut property 用交换论证刻画安全边"
  - "Minimum spanning tree cut property"
  - "MST cut property"
student_os: knowledge-atom
atom_id: MCS-GRAPH-030
atom_set: acyclic-graphs-trees-dags
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[有限树的等价刻画]]"
  - "[[最小生成树的定义与唯一性]]"
leads_to:
  - "[[Kruskal 算法]]"
  - "[[Prim 算法]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

# MST cut property 用交换论证刻画安全边
<!-- bilingual-en:start -->
*The MST cut property characterizes safe edges by an exchange argument*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 在 connected weighted undirected graph 中，取任意非平凡 cut $(S,V\setminus S)$。一条 minimum-weight crossing edge $e$ 属于**某个** MST；若 $e$ 是该 cut 上唯一最轻的 crossing edge，则它属于**每个** MST。
> <!-- bilingual-en:start -->
> In a connected weighted undirected graph, take any nontrivial cut $(S,V\setminus S)$. A minimum-weight edge $e$ crossing the cut belongs to **some** MST. If $e$ is the unique lightest crossing edge, it belongs to **every** MST.
> <!-- bilingual-en:end -->

证明采用 exchange。取不含 $e$ 的 MST $T$，把 $e$ 加入后，tree 的唯一 path 性保证恰产生一个 cycle。该 cycle 用 $e$ 从 cut 一侧进入另一侧，必还要用某条 $f\in T$ 跨回来。删 $f$ 得 spanning tree $T'=T+e-f$，而 $w(e)\le w(f)$，所以 $w(T')\le w(T)$；由 $T$ 已最优，$T'$ 也是含 $e$ 的 MST。若 $e$ 唯一最轻，则任何遗漏它的 MST 都会被严格改进，矛盾。
<!-- bilingual-en:start -->
Take an MST $T$ omitting $e$. Adding $e$ creates exactly one cycle. Because $e$ crosses the cut, that cycle contains another crossing edge $f\in T$. Then $T'=T+e-f$ is a spanning tree and $w(e)\le w(f)$, so $w(T')\le w(T)$. Optimality makes $T'$ another MST. If $e$ is uniquely lightest, the exchange would strictly improve every tree omitting it.
<!-- bilingual-en:end -->

算法中的“safe edge”还带一个状态条件。若当前已选边集 $A$ 已包含在某个 MST 中，要保证加入 light edge 后仍可扩展，所用 cut 必须 **respect $A$**：$A$ 中没有边跨越该 cut。否则 exchange 时被删的 $f$ 可能本来就在 $A$ 中，不能证明保留所有既有选择。
<!-- bilingual-en:start -->
The algorithmic safe-edge form has an additional state condition. If the current set $A$ is contained in some MST, the cut must **respect $A$**, meaning no edge of $A$ crosses it. Otherwise the exchange edge $f$ might itself belong to $A$, so the proof would not preserve all earlier choices.
<!-- bilingual-en:end -->

> [!question]- 自检
> cut 上有两条并列最轻边，能否说二者都属于每个 MST？
>
> **答案：**不能。cut property 只保证每一条最轻边分别属于某个 MST；只有“唯一最轻”才能推出属于每个 MST。

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=541|MIT Mathematics for Computer Science, Gray Edge Lemma 12.11.11]]：核对 distinct-weight 情形的 cut/gray-edge 结论与交换结构。
- [MIT 6.046J Lecture 4, Minimum Spanning Trees II](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2012/bb54568c86c60d86b25e9c1111470384_MIT6_046JS12_lec04.pdf)：核对允许 ties 时的 light edge、respecting cut 与 safe edge 精确定式。

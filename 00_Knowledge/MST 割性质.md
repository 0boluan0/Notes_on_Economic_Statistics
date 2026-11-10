---
aliases:
  - Minimum spanning tree cut property
  - MST cut property
  - 最小生成树割性质
student_os: knowledge-atom
atom_id: MCS-GRAPH-030
atom_set: acyclic-graphs-trees-dags
atom_type: theorem
status: source-checked
requires:
  - "[[图割]]"
  - "[[有限树的等价刻画]]"
  - "[[最小生成树的定义与唯一性]]"
leads_to:
  - "[[Kruskal 算法]]"
  - "[[Prim 算法]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

<!-- bilingual-en:start -->
# The MST cut property
<!-- bilingual-en:end -->
*MST 割性质*

<!-- bilingual-en:start -->
The **cut property** explains why a locally cheap connection can be compatible with a globally minimum spanning tree. Split the vertices into two nonempty sides. Every spanning tree must connect those sides somewhere. A cheapest edge across that particular split can be included in some MST; when it is the only cheapest crossing edge, every MST must include it.
<!-- bilingual-en:end -->
**割性质**解释为什么局部便宜的连接可以与全局最小生成树兼容。把顶点分成两个非空部分，每棵生成树都必须在某处连接这两边。跨越这次划分的最便宜边，可以属于某棵 MST；若它是唯一最便宜的跨割边，则每棵 MST 都必须包含它。

<!-- bilingual-en:start -->
## A cut and its light edges
<!-- bilingual-en:end -->
*割与最轻跨割边*

<!-- bilingual-en:start -->
Let $G=(V,E,w)$ be a finite, nonempty, connected undirected graph with finite real edge weights. A [[最小生成树的定义与唯一性|minimum spanning tree]] minimizes the total edge weight among all spanning trees. Choose a nonempty proper subset $S$ of $V$. The two sides $(S,V\setminus S)$ form a **[[图割|cut]]**, where $V\setminus S$ means the vertices outside $S$. An edge **crosses** the cut when exactly one endpoint is in $S$. Write
<!-- bilingual-en:end -->
设 $G=(V,E,w)$ 是有限、非空、连通的无向图，边权是有限实数。[[最小生成树的定义与唯一性|最小生成树]]在全部生成树中最小化边权总和。取 $V$ 的非空真子集 $S$，两边 $(S,V\setminus S)$ 构成一个**[[图割|割]]**，其中 $V\setminus S$ 表示不在 $S$ 中的顶点。恰有一个端点在 $S$ 中的边称为**跨割边**。记

$$\delta(S)=\{e\in E:e\text{ has exactly one endpoint in }S\}.$$

<!-- bilingual-en:start -->
Connectedness guarantees that $\delta(S)$ is nonempty: a path from one side to the other must cross it. A crossing edge $e\in\delta(S)$ is **light** for this cut when $w(e)\le w(f)$ for every $f\in\delta(S)$. “Light” compares crossing edges of this cut, not all edges of the graph. The finite set of crossing edges has a minimum, and ties are allowed.
<!-- bilingual-en:end -->
连通性保证 $\delta(S)$ 非空，因为从一边到另一边的路径必然经过跨割边。若跨割边 $e\in\delta(S)$ 对每条 $f\in\delta(S)$ 都满足 $w(e)\le w(f)$，就称为这个割的一条**最轻边**。这里比较的是这个割上的边，不是全图所有边。跨割边有限且非空，所以有最小权重，允许并列。

> [!important] Some MST or every MST? · 某棵还是每棵？
> <!-- bilingual-en:start -->
> Every light edge of a cut belongs to **some** MST. If it is the **unique** light edge of that cut, it belongs to **every** MST. Uniqueness is required only among the edges crossing this cut, not among all weights in the graph.
> <!-- bilingual-en:end -->
>
> 割上的每一条最轻边都属于**某棵** MST。如果它是这个割上**唯一**最轻的边，则属于**每棵** MST。这里的唯一性只比较跨越这个割的边，不要求全图权重互异。

<!-- bilingual-en:start -->
## Exchange one edge without increasing the weight
<!-- bilingual-en:end -->
*交换一条边而不增加总权重*

<!-- bilingual-en:start -->
Take any MST $T$. If it already contains $e$, the “some MST” claim is satisfied. Otherwise let $u,v$ be the endpoints of $e$. The [[有限树的等价刻画|unique path in the tree]] from $u$ to $v$, together with the new edge $e$, forms exactly one cycle. Since $u,v$ are on opposite sides of the cut, that path includes another crossing edge $f\in T$.
<!-- bilingual-en:end -->
任取一棵 MST $T$。若它已经包含 $e$，“属于某棵”的结论已经成立。否则，设 $e$ 的端点为 $u,v$。连接 $u,v$ 的[[有限树的等价刻画|唯一树路径]]加上新边 $e$，恰好形成一个环。因为 $u,v$ 在割的两侧，这条路径上一定还有另一条跨割边 $f\in T$。

<!-- bilingual-en:start -->
Delete $f$ after adding $e$, giving the edge set of $T'$:
<!-- bilingual-en:end -->
加入 $e$ 后再删去 $f$，得到新树 $T'$ 的边集：

$$E(T')=(E(T)\cup\{e\})\setminus\{f\}.$$

<!-- bilingual-en:start -->
Removing a cycle edge preserves connectedness. It also removes the only cycle, so $T'$ is again a spanning tree. Because $e$ is light across the cut,
<!-- bilingual-en:end -->
删去环边保持连通，又消除了唯一的环，因此 $T'$ 仍是生成树。由于 $e$ 是最轻跨割边，

$$w(T')=w(T)+w(e)-w(f)\le w(T).$$

<!-- bilingual-en:start -->
But $T$ already has minimum weight, so equality holds. Thus $T'$ is an MST containing $e$. If $e$ is uniquely light and $T$ omits it, then $f\ne e$ gives $w(e)<w(f)$; the same exchange would produce a strictly lighter tree, a contradiction. Therefore no MST can omit a uniquely light crossing edge. The proof compares weights only, so zero and negative values cause no difficulty.
<!-- bilingual-en:end -->
但 $T$ 已经达到最小权重，所以只能取等号。于是 $T'$ 是包含 $e$ 的 MST。若 $e$ 唯一最轻而 $T$ 不含它，则 $f\ne e$ 给出 $w(e)<w(f)$；同样的交换会得到严格更轻的树，矛盾。因此，任何 MST 都不能遗漏唯一最轻的跨割边。证明只比较权重大小，零权和负权不会造成困难。

> [!example] Tied light edges need not occur in every MST · 并列最轻不表示每棵都包含
> <!-- bilingual-en:start -->
> Give all three edges of the triangle $a,b,c$ weight $1$, and take $S=\{a\}$. Both $ab$ and $ac$ are light crossing edges. The tree $\{ac,bc\}$ omits $ab$, while $\{ab,bc\}$ omits $ac$; both have the minimum total $2$. Each light edge belongs to some MST, but neither belongs to every MST. If instead the weights are $w(ab)=1,w(ac)=3,w(bc)=2$, then $ab$ is uniquely light across this cut and must occur in every MST.
> <!-- bilingual-en:end -->
>
> 三角形 $a,b,c$ 的三条边权全为 $1$，取 $S=\{a\}$。$ab,ac$ 都是最轻跨割边。树 $\{ac,bc\}$ 不含 $ab$，树 $\{ab,bc\}$ 不含 $ac$，两者总权重都达到最小值 $2$。每条最轻边都属于某棵 MST，但都不是每棵必含。若改为 $w(ab)=1,w(ac)=3,w(bc)=2$，则 $ab$ 是这个割上唯一最轻的边，每棵 MST 都必须包含它。

<!-- bilingual-en:start -->
## Preserve the choices already made
<!-- bilingual-en:end -->
*同时保留已经作出的选择*

<!-- bilingual-en:start -->
An algorithm needs more than “this edge occurs in some MST”: the new edge must be compatible with **all edges already selected**. Let $A\subseteq E$ be the selected edge set, and assume there is an MST $T$ containing the whole set $A$. An edge $e\notin A$ is **safe for $A$** if some MST contains $A\cup\{e\}$. A cut **respects $A$** when none of the selected edges crosses it, that is, $A\cap\delta(S)=\varnothing$.
<!-- bilingual-en:end -->
算法需要的不只是“这条边出现在某棵 MST 中”，而是它与**此前选中的全部边**兼容。设 $A\subseteq E$ 为已选边集，并假设存在一棵 MST $T$ 包含整个 $A$。若某棵 MST 包含 $A\cup\{e\}$，则尚未选中的边 $e\notin A$ 称为**对 $A$ 安全**。若已选边没有一条跨越某个割，即 $A\cap\delta(S)=\varnothing$，就说这个割**尊重 $A$**。

<!-- bilingual-en:start -->
If a cut respects $A$ and $e$ is light across it, then $e$ is safe for $A$. To prove this, start the exchange argument with an MST containing $A$. If it contains $e$ too, nothing needs changing. Otherwise the removed edge $f$ crosses the cut, so $f\notin A$. The exchange therefore keeps every selected edge while adding $e$, and the resulting tree is still an MST. Respecting the cut is a sufficient condition for this guarantee, not a claim that every safe edge must arise from any particular proposed cut.
<!-- bilingual-en:end -->
如果割尊重 $A$，且 $e$ 是该割上的最轻边，那么 $e$ 对 $A$ 安全。证明时，从包含 $A$ 的一棵 MST 开始交换。若它也含 $e$，无需修改；否则，被删的 $f$ 跨割，所以 $f\notin A$。因此交换既加入了 $e$，又保留了每条已选边，结果仍是 MST。尊重条件给出这项保证的一条充分路线，并不是说每条安全边都必须由任意指定的某个割来证明。

<!-- bilingual-en:start -->
In the all-weight-$1$ triangle, take $A=\{ab,bc\}$ and propose $e=ac$. The set $A$ is already an MST, and $e$ is light for the cut $\{a\}\mid\{b,c\}$, but that cut does not respect $A$ because $ab$ crosses it. Adding $ac$ to $A$ makes a cycle, so no spanning tree can contain all three edges. This shows why independently valid choices cannot simply be accumulated.
<!-- bilingual-en:end -->
在三条边权全为 $1$ 的三角形中，取 $A=\{ab,bc\}$，想再选 $e=ac$。$A$ 已是一棵 MST，$e$ 也是割 $\{a\}\mid\{b,c\}$ 上的最轻边，但这个割不尊重 $A$，因为 $ab$ 跨割。把 $ac$ 加入 $A$ 就成环，没有生成树能同时包含三条边。这说明，单独看来合法的选择，不能不加条件地累积起来。

<!-- bilingual-en:start -->
[[Kruskal 算法|Kruskal's algorithm]] uses cuts defined by current forest components; [[Prim 算法|Prim's algorithm]] uses the current tree's vertices against the outside. Their correctness arguments establish both the compatible-selected-set invariant and the appropriate light-edge condition. A singleton graph has no nontrivial cut to check. Self-loops never cross a cut; parallel crossing edges are compared as distinct edges and may tie. These conventions agree with the tree and MST definitions.
<!-- bilingual-en:end -->
[[Kruskal 算法]]使用当前森林分量与其余顶点构成的割，[[Prim 算法]]使用当前树顶点与外部构成的割。它们的正确性论证要同时建立“已选边仍共同兼容某棵 MST”的不变量，以及相应的最轻边条件。单点图没有非平凡割可检查。自环永不跨割；平行跨割边按不同边比较，可以并列。这些约定与树和 MST 的定义一致。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [MIT6.046J Lecture3, PDF5–6](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2012/6baf48ebe3babed294c31a907a916e08_MIT6_046JS12_lec03.pdf#page=5): Theorem3.3 and its exchange proof, including ties; the strict exchange gives the uniquely-light conclusion.
- [MIT6.046J Lecture4, PDF4, Proposition4.1](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2012/bb54568c86c60d86b25e9c1111470384_MIT6_046JS12_lec04.pdf#page=4): safe selected sets, respecting cuts, and retaining every previously selected edge during the exchange.
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=541|MIT Mathematics for Computer Science, PDF541–542]], Lemma12.11.11 and the gray-edge procedure: the course's cut interpretation and component growth. Its “every MST” statement is under the explicitly preceding distinct-weight assumption; the tie-allowing version is proved above.
- [[01_Math/07-Mathematics for Computer Science/02_Structures.md#21.4 cut/gray-edge lemma|Permanent Session21.4]]: course notation and the some/every distinction, followed by the exchange calculation.
<!-- bilingual-en:end -->

- MIT6.046J Lecture3，PDF5–6：定理3.3及允许并列的交换证明；严格交换给出唯一最轻边的必含结论。
- MIT6.046J Lecture4，PDF4，命题4.1：安全已选集、尊重割，以及交换时保留全部已选边。
- MIT《Mathematics for Computer Science》，PDF541–542，Lemma12.11.11与灰边过程：课程的割解释及分量生长；其中“每棵 MST”使用此前明确的互异权重假设，允许并列的版本在正文单独证明。
- 永久 Session21.4：课程记号、某棵与每棵的区分，以及交换的权重计算。

---
aliases:
  - Forest edge-count formula
  - Forest 的 n-c 边公式
student_os: knowledge-atom
atom_id: MCS-GRAPH-025
atom_set: acyclic-graphs-trees-dags
atom_type: theorem
status: source-checked
requires:
  - "[[有限树的等价刻画]]"
  - "[[无向图连通分量]]"
related:
  - "[[生成树的存在条件]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

<!-- bilingual-en:start -->
# Counting edges in a forest
<!-- bilingual-en:end -->

*森林的边数公式*

<!-- bilingual-en:start -->
In a forest, every edge does useful connecting work: it joins two pieces that were previously separate. There is no cycle edge that merely supplies another route inside an already connected piece. Counting those mergers gives a direct relation between the number of vertices, edges and connected components.
<!-- bilingual-en:end -->

森林中的每条边都承担一次有效连接：它把原先分离的两部分接起来，没有哪条环边只是在已连通部分内部多提供一条路线。数清这些合并次数，就能得到顶点数、边数与连通分量数的关系。

<!-- bilingual-en:start -->
A finite [[森林|forest]] is an acyclic undirected graph. Each of its [[无向图连通分量|connected components]] is an [[无向树|undirected tree]], and an isolated vertex counts as a component. Let $n=|V|$ be the number of vertices, $m=|E|$ the number of edges, and $c$ the number of components of the forest itself.
<!-- bilingual-en:end -->

有限[[森林]]是不含环的无向图，每个[[无向图连通分量]]都是一棵[[无向树]]，孤立顶点也算一个分量。记 $n=|V|$ 为顶点数，$m=|E|$ 为边数，$c$ 为这张森林自身的分量数。

> [!important] One edge per merger · 每次合并对应一条边
> <!-- bilingual-en:start -->
> Every finite forest satisfies
> <!-- bilingual-en:end -->
>
> $$|E|=n-c.$$
>
> <!-- bilingual-en:start -->
> In particular, a tree has $c=1$, so a tree on $n$ vertices has exactly $n-1$ edges. The empty forest has $n=m=c=0$, and also satisfies the formula.
> <!-- bilingual-en:end -->
>
> 每张有限森林均满足上述公式。树只有一个分量，即 $c=1$，所以有 $n$ 个顶点的树恰有 $n-1$ 条边。空森林满足 $n=m=c=0$，公式同样成立。

<!-- bilingual-en:start -->
## Build the forest one edge at a time
<!-- bilingual-en:end -->

*逐条加入边，数分量的变化*

<!-- bilingual-en:start -->
Keep all $n$ vertices and initially select no edges. There are then $n$ isolated components. Add the edges of the given forest in any order. Before an edge is added, its two endpoints must belong to different current components. Otherwise an existing [[游走路径与环|path]] between them, followed by the new edge, would form a cycle contained in the final forest. This is impossible.
<!-- bilingual-en:end -->

先保留全部 $n$ 个顶点，但不选任何边，此时有 $n$ 个孤立分量。按任意次序加入给定森林的各条边。每次加入前，这条边的两个端点必须属于两个不同的当前分量；否则两端之间已有[[游走路径与环|路径]]，再加新边就形成最终森林中的环，矛盾。

<!-- bilingual-en:start -->
An edge joining two different components merges exactly those two components into one. It does not change any other component. Thus every added edge decreases the component count by exactly one. After all $m$ edges have been added,
<!-- bilingual-en:end -->

连接两个不同分量的边，恰好把这两个分量合成一个，不会改变其他分量。因此每加入一条边，分量数恰好减少一。加入全部 $m$ 条边后，

$$c=n-m,$$

<!-- bilingual-en:start -->
which rearranges to $m=n-c$. This proof does not assume the tree edge formula in advance and does not depend on a particular edge order. With no vertices or edges there are no steps to perform, and the equality is $0=0$.
<!-- bilingual-en:end -->

移项即得 $m=n-c$。这个证明没有预先借用树的边数公式，也不依赖选边次序。没有顶点或边时无需进行任何步骤，等式就是 $0=0$。

> [!example] Count isolated vertices too · 孤立顶点也要计数
> <!-- bilingual-en:start -->
> Take $a,b,c,d,e$ with only the edges $\{a,b\}$ and $\{c,d\}$. Starting from five components, the first edge leaves four and the second leaves three. The final components are $\{a,b\}$, $\{c,d\}$ and $\{e\}$. Hence $m=2=5-3$. More generally, a forest with 20 vertices and four components has $20-4=16$ edges; knowing only that some graph has 16 edges says nothing by itself about acyclicity.
> <!-- bilingual-en:end -->
>
> 取顶点 $a,b,c,d,e$，只有边 $\{a,b\}$、$\{c,d\}$。起初五个分量，第一条边使它剩四个，第二条使它剩三个；最终分量是 $\{a,b\}$、$\{c,d\}$、$\{e\}$，因此 $m=2=5-3$。同理，20 个顶点、四个分量的森林有 $20-4=16$ 条边；只知道某图有 16 条边，本身不能判断它是否无环。

<!-- bilingual-en:start -->
## When the count identifies a tree
<!-- bilingual-en:end -->

*什么时候边数可以用来识别树*

<!-- bilingual-en:start -->
The number $n-1$ alone is insufficient. A triangle together with an isolated vertex has $n=4$ and $m=3=n-1$, but it is both disconnected and cyclic. For a finite nonempty simple undirected graph, the count becomes a tree criterion when combined with either of two structural assumptions.
<!-- bilingual-en:end -->

单独的 $n-1$ 不够。一个三角形再加一个孤立顶点，满足 $n=4$、$m=3=n-1$，却既不连通又有环。对有限非空简单无向图，边数还要结合以下两种结构前提之一，才能作为树的判据。

<!-- bilingual-en:start -->
If the graph is **acyclic** and has $n-1$ edges, it is a forest, so $n-c=n-1$ gives $c=1$. It is connected and hence a tree. If the graph is **connected** and has $n-1$ edges, repeatedly remove an edge from a cycle while a cycle remains. Its endpoints still have the rest of that cycle as a detour, so connectedness is preserved. Finitely many deletions leave a tree on the same $n$ vertices. The proved forest formula says that this final tree has $n-1$ edges already. Since the starting graph had only $n-1$ edges, no deletion could have occurred; the starting graph was a tree.
<!-- bilingual-en:end -->

若图**无环**且有 $n-1$ 条边，它就是森林，由 $n-c=n-1$ 得 $c=1$，所以连通，是树。若图**连通**且有 $n-1$ 条边，则只要还存在环，就删去环上的一条边；两端仍能沿环的其余部分绕行，连通性保持。有限次删除后得到一棵顶点仍为 $n$ 个的树。已经证明的森林公式说明，这棵最终的树本身就有 $n-1$ 条边；起点也只有这么多边，因此实际上不可能删去过任何一条，原图本来就是树。

<!-- bilingual-en:start -->
This argument explains the numerical counterpart of the [[有限树的等价刻画|structural tree tests]]. It uses no assumption that every graph with $n-1$ edges is connected.
<!-- bilingual-en:end -->

这就是[[有限树的等价刻画|树的结构检验]]在边数上的对应说法；证明没有偷用“任意 $n-1$ 边图都连通”这一错误断言。

<!-- bilingual-en:start -->
## Count the selected graph's components
<!-- bilingual-en:end -->

*数所选图自己的分量*

<!-- bilingual-en:start -->
If a forest is selected inside a larger host graph, $c$ means the selected forest's component count. For the host path $a-b-c$, selecting only $\{a,b\}$ gives $n=3$, $c=2$, $m=1$. The host itself has one component, but substituting that count would be wrong. A [[森林|spanning forest in the componentwise sense]] keeps one spanning tree per host component, so only then do the two component counts coincide. Self-loops and parallel two-cycles cannot occur in a forest under the stated cycle convention. All counts here are finite; the subtraction is not a formula for arbitrary infinite cardinalities.
<!-- bilingual-en:end -->

若从一张较大的原图中选出森林，$c$ 指所选森林自己的分量数。例如，原图为路径 $a-b-c$，只选 $\{a,b\}$ 时，有 $n=3,c=2,m=1$。原图虽只有一个分量，却不能把这个数代入。只有[[森林|按原分量生成的森林]]在每个原分量内选出一棵生成树，两个分量数才相同。按本库的环约定，森林中不会出现自环或平行二环。这里所有计数都是有限数，不能把减法直接解释为任意无限基数的公式。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=537|MIT Mathematics for Computer Science, PDF 537–538]], Theorems 12.11.5–12.11.6: forest component count and the two finite $n-1$ characterizations. The text uses leaf-removal induction; the edge-merging proof above proves the same count directly, including the explicit empty case.
- [[01_Math/07-Mathematics for Computer Science/02_Structures.md#21.2 forest、leaves 与二着色|Course Session 21.2]]: the course's $n-c$ notation and isolated-component interpretation.
- [Richard Stanley, Topics in Algebraic Combinatorics, PDF 135 and 170](https://math.mit.edu/~rstan/algcomb/algcomb.pdf#page=135), Proposition 9.1 and §11.2: finite tree edge characterizations and maximal forests that span each host component.
<!-- bilingual-en:end -->

- MIT《Mathematics for Computer Science》，PDF 537–538，定理 12.11.5–12.11.6：核对森林分量数公式，以及两种有限图的 $n-1$ 刻画。教材用删叶归纳，正文用逐边合并直接证明同一结论，并明确空森林情形。
- 课程 Session 21.2：核对 $n-c$ 的记号和孤立分量解释。
- Richard Stanley《Topics in Algebraic Combinatorics》，PDF 135、170，命题 9.1 及 §11.2：核对有限树的边数刻画，以及在原图每个分量内生成的极大森林。

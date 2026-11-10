---
aliases:
  - Trees and DAGs are different acyclic structures
  - 无向树与有向无环图的区别
student_os: knowledge-atom
atom_id: MCS-GRAPH-023
atom_set: acyclic-graphs-trees-dags
atom_type: distinction
status: source-checked
requires:
  - "[[图类型的建模边界]]"
  - "[[游走路径与环]]"
leads_to:
  - "[[有限树的等价刻画]]"
  - "[[DAG 与拓扑序等价]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

<!-- bilingual-en:start -->
# Trees and DAGs
<!-- bilingual-en:end -->

*无向树与有向无环图的区别*

<!-- bilingual-en:start -->
A tree and a directed acyclic graph both exclude cycles, but they exclude different kinds of routes. A tree describes undirected connections with no redundant route. A DAG describes arrows that never lead around a directed cycle. Before transferring a statement about one to the other, decide whether a route may ignore the arrows.
<!-- bilingual-en:end -->

树和有向无环图都排除环，但排除的是不同类型的路线。树描述没有冗余路线的无向连接；DAG 描述不会沿方向绕成环的箭头。把一类图的结论用于另一类之前，要先确定路线能不能忽略箭头。

<!-- bilingual-en:start -->
All graphs in this card are finite. An [[无向树|undirected tree]] is a nonempty, connected, acyclic undirected graph; it has no self-loops or parallel edges. A [[有向无环图|directed acyclic graph]], abbreviated **DAG**, has no directed cycle, meaning a cycle that follows every arrow from tail to head. A DAG does not have to be connected. These definitions use the [[游走路径与环|cycle convention]] that excludes traversing one undirected edge out and back as a cycle.
<!-- bilingual-en:end -->

本卡讨论有限图。[[无向树]]是非空、连通且无环的无向图，没有自环或平行边。[[有向无环图]]简称 **DAG**，不含有向环，也就是不能沿每条边从起点到终点绕成一圈；DAG 不要求连通。这些定义采用本库的[[游走路径与环|环约定]]：沿同一条无向边走出去再原路返回，不算环。

<!-- bilingual-en:start -->
## Giving a tree one arrow per edge
<!-- bilingual-en:end -->

*给树的每条边指定一个方向*

<!-- bilingual-en:start -->
Take the undirected tree $a-b-c$. Choose one direction for each edge, for example $a\to b$ and $c\to b$. There is no directed cycle. More generally, **every orientation of a tree is a DAG**, where an orientation replaces each undirected edge by exactly one of its two possible directed edges. If a directed cycle appeared, forgetting its arrows would give an undirected cycle in the original tree, which is impossible.
<!-- bilingual-en:end -->

取无向树 $a-b-c$，给每条边选择一个方向，例如 $a\to b$、$c\to b$，得到的图没有有向环。一般地，**树的任意定向都是 DAG**；这里“定向”是把每条无向边替换为两个可能方向中的恰好一个。若出现有向环，去掉这些箭头后，就在原树中得到无向环，矛盾。

<!-- bilingual-en:start -->
The words “one direction” matter. Replacing the single tree edge $a-b$ by both $a\to b$ and $b\to a$ creates two directed edges and a directed two-cycle. That operation is not an orientation of the original edge. Similarly, merely storing an undirected edge twice in an adjacency list does not turn its two records into two graph edges; the [[图类型的建模边界|graph model]] determines what the records mean.
<!-- bilingual-en:end -->

“一个方向”不能省略。把树的一条边 $a-b$ 同时替换为 $a\to b$、$b\to a$，便产生两条有向边和有向二环；这不是对原边做一次定向。同样，在邻接表中把一条无向边记录两次，不会自动把两条记录变成两条图边；具体含义由[[图类型的建模边界|图模型]]决定。

<!-- bilingual-en:start -->
An orientation also need not describe a hierarchy with one vertex able to reach all the others. In $a\to b\leftarrow c$, neither $a$ nor $c$ can reach the other, and $b$ cannot leave. It is still a DAG whose underlying undirected graph is a tree. Choosing a root and directing every edge away from it would be an additional choice.
<!-- bilingual-en:end -->

任意定向也不一定形成“从某个顶点出发就能到达全部其他顶点”的层级结构。在 $a\to b\leftarrow c$ 中，$a,c$ 互不可达，$b$ 又无边可走；它仍是底层无向图为树的 DAG。另选一个根，并把全部边从根向外定向，属于额外选择。

<!-- bilingual-en:start -->
## A DAG can retain an undirected cycle
<!-- bilingual-en:end -->

*DAG 的底层无向图仍然可能有环*

> [!example] Two routes through a diamond · 菱形图中的两条路线
> <!-- bilingual-en:start -->
> Consider four vertices with the arrows
> <!-- bilingual-en:end -->
>
> $$A\to B,\qquad A\to C,\qquad B\to D,\qquad C\to D.$$
>
> <!-- bilingual-en:start -->
> All arrows move from $A$, through $B$ or $C$, toward $D$. There is no way to return to an earlier vertex, so this is a DAG. Yet $A$ reaches $D$ by both $A\to B\to D$ and $A\to C\to D$. Forgetting directions leaves the undirected cycle $A-B-D-C-A$. Thus the underlying graph is not a tree. It has four vertices and four edges, rather than the three edges a four-vertex tree would have.
> <!-- bilingual-en:end -->
>
> 所有箭头都从 $A$ 经 $B$ 或 $C$ 走向 $D$，没有返回上游顶点的路线，所以它是 DAG。但从 $A$ 到 $D$ 既可走 $A\to B\to D$，也可走 $A\to C\to D$。去掉方向后，留下无向环 $A-B-D-C-A$，底层图因此不是树。它有四个顶点、四条边，而四顶点树只有三条边。

<!-- bilingual-en:start -->
This example disproves both “a DAG must become a tree when arrows are erased” and “a DAG has at most one directed path between two vertices.” Disconnectedness is a separate issue: two isolated vertices already form a DAG, but not a tree. If the empty graph is allowed, it is a DAG and is excluded by the nonempty tree definition; a single isolated vertex is a tree and gives a DAG with no arrows.
<!-- bilingual-en:end -->

这个例子同时否定“DAG 去掉箭头后必是树”和“DAG 中两点间至多有一条有向路径”。连通性又是另一个问题：两个孤立顶点已经构成 DAG，却不是树。若允许空图，它是 DAG，但被树的非空定义排除；单个孤立顶点则是一棵树，也给出一张没有箭头的 DAG。

<!-- bilingual-en:start -->
## Use the conclusion belonging to the stated structure
<!-- bilingual-en:end -->

*按结构前提选择结论*

<!-- bilingual-en:start -->
For a tree, [[有限树的等价刻画|unique undirected paths and disconnection after deleting any edge]] follow from connectedness and undirected acyclicity. The [[Forest 的边数公式|finite edge-count theorem]] then gives $n-1$ edges when there are $n$ vertices. These are not consequences of directed acyclicity alone.
<!-- bilingual-en:end -->

对树，连通且无向无环推出[[有限树的等价刻画|无向路径唯一、删去任意边就断开]]，再由[[Forest 的边数公式|有限边数定理]]得到 $n$ 个顶点对应 $n-1$ 条边。这些都不是单凭“没有有向环”就能得到的结论。

<!-- bilingual-en:start -->
For a finite DAG, the relevant ordering result is that a [[拓扑序|topological order]] exists: every vertex is listed once and every arrow points forward. That [[DAG 与拓扑序等价|existence theorem]] permits multiple routes and disconnected graphs. When an arrow means “prerequisite first,” it supports dependency ordering; it does not turn the graph into a tree or specify a unique schedule. Neither an undirected drawing nor a reference arrow used only for matrix signs should be silently reinterpreted as such a dependency model.
<!-- bilingual-en:end -->

对有限 DAG，相应的顺序结论是存在[[拓扑序]]：把每个顶点恰好列一次，并让每条箭头都朝向列表后方。这个[[DAG 与拓扑序等价|存在性定理]]允许多条路线，也允许不连通图。当箭头表示“前置任务在先”时，它支持依赖排序，却不会把图变成树，也没有指定唯一调度。不能把无向图的画法，或只为矩阵符号服务的参考箭头，默默解释为这种依赖模型。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=439|MIT Mathematics for Computer Science, PDF 439–443]], Definition 10.5.1 and §10.5.1: DAGs, directed prerequisites and topological orders. The orientation, diamond and boundary examples above are checked directly from these definitions.
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=535|The same textbook, PDF 535–536]], Definition 12.11.1 and Theorem 12.11.4: trees in the undirected simple-graph setting and their unique-path structure.
- [[01_Math/07-Mathematics for Computer Science/02_Structures.md#Session 17 — Directed Acyclic Graphs|Course Session 17]] and [[01_Math/07-Mathematics for Computer Science/02_Structures.md#Session 21 — Trees and Minimum Spanning Trees|Session 21]]: the course's directed versus undirected terminology and applications.
<!-- bilingual-en:end -->

- MIT《Mathematics for Computer Science》，PDF 439–443，定义 10.5.1 及 §10.5.1：核对 DAG、有向先修关系与拓扑序。正文的定向、菱形及边界例子直接按定义核验。
- 同书 PDF 535–536，定义 12.11.1 及定理 12.11.4：核对简单无向图中的树及唯一路径结构。
- 课程 Session 17、Session 21：核对课程的有向、无向术语与应用分工。

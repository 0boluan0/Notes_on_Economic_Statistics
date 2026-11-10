---
aliases:
  - Proper vertex coloring
  - Proper vertex coloring and independent-set partition
  - 冲突图着色
student_os: knowledge-atom
atom_id: MCS-COLOR-001
atom_type: definition
status: source-checked
part_of:
  - "[[图着色与色数.canvas]]"
requires:
  - "[[图模型语义]]"
implies:
  - "[[精确色数的上下界]]"
---

<!-- bilingual-en:start -->
# Proper vertex coloring
<!-- bilingual-en:end -->

*合法顶点着色与独立集划分*

<!-- bilingual-en:start -->
A **proper vertex coloring** assigns a label to every vertex so that the endpoints of every edge receive different labels. Think of three exams $a,b,c$, with conflicts only between $a,b$ and between $b,c$. Put $a,c$ in time slot 1 and $b$ in slot 2. Every conflict is separated even though two exams share a slot. The colors name groups; they need not be literal colors.
<!-- bilingual-en:end -->

一次**合法顶点着色**给每个顶点分配标签，并使每条边的两端标签不同。设三门考试 $a,b,c$ 仅在 $a,b$ 之间以及 $b,c$ 之间冲突。把 $a,c$ 放在时段1，把 $b$ 放在时段2，就分开了每一对冲突，虽然仍有两门考试同场。颜色是组的标签，不必真的是色彩。

<!-- bilingual-en:start -->
## What a k-coloring requires
<!-- bilingual-en:end -->

*k 着色要求什么*

<!-- bilingual-en:start -->
Let $G=(V,E)$ be a finite [[图类型的建模边界|simple undirected graph]], and let $k$ be a positive integer. A **proper $k$-coloring** is a function $c$ that assigns one of the labels $1,\ldots,k$ to each vertex:
<!-- bilingual-en:end -->

设 $G=(V,E)$ 是有限[[图类型的建模边界|简单无向图]]，$k$ 是正整数。**合法 $k$ 着色**是函数 $c$，它给每个顶点分配 $1,\ldots,k$ 中的一个标签：

$$c:V\to\{1,\ldots,k\},\qquad \{u,v\}\in E\Rightarrow c(u)\ne c(v).$$

<!-- bilingual-en:start -->
The function assigns every vertex exactly one label. All $k$ labels need not be used: a proper coloring using two labels is also a proper 3-coloring after viewing its labels as belonging to $\{1,2,3\}$. “$k$-colorable” means that such a coloring exists. It does not say $k$ is the smallest possible number; that minimum is the [[色数|chromatic number]].
<!-- bilingual-en:end -->

这个函数给每个顶点恰好一个标签，不要求 $k$ 个标签全都出现：只使用两个标签的合法着色，也可以看成标签取自 $\{1,2,3\}$ 的合法三着色。“可 $k$ 着色”表示至少存在一种这样的着色，不表示 $k$ 已经最小；最小所需数叫作[[色数]]。

<!-- bilingual-en:start -->
## Color classes give a partition
<!-- bilingual-en:end -->

*颜色类为什么给出划分*

<!-- bilingual-en:start -->
For a label $i$, its **color class** is $C_i=\{v\in V:c(v)=i\}$. Any two vertices in $C_i$ must be nonadjacent, so $C_i$ is an [[图的独立集|independent vertex set]]. Distinct classes are disjoint, and together they cover $V$. Discard unused, empty classes: the remaining sets form a partition of $V$ into at most $k$ nonempty independent sets.
<!-- bilingual-en:end -->

标签 $i$ 的**颜色类**是 $C_i=\{v\in V:c(v)=i\}$。其中任意两点不能相邻，所以 $C_i$ 是[[图的独立集]]。不同颜色类互不相交，合起来覆盖 $V$。去掉未使用的空颜色类后，剩下的集合把 $V$ 划分成至多 $k$ 个非空独立集。

<!-- bilingual-en:start -->
Conversely, suppose disjoint nonempty sets $C_1,\ldots,C_r$ cover $V$, each is independent, and $r\le k$. Give every vertex in $C_i$ the label $i$. If an edge had equally colored endpoints, both would belong to one independent set, a contradiction. This proves the equivalence in both directions. For an empty vertex set, the empty family is its partition and the empty function is a proper $k$-coloring.
<!-- bilingual-en:end -->

反过来，设互不相交的非空集合 $C_1,\ldots,C_r$ 覆盖 $V$，每一组都是独立集，而且 $r\le k$。给 $C_i$ 中的每个顶点标上 $i$。若一条边两端同色，就意味着两端都在同一个独立集内，矛盾。于是两方向都成立。顶点集合为空时，用空集合族作为划分，空函数就是合法 $k$ 着色。

<!-- bilingual-en:start -->
Colors are only group labels. Swapping the names “red” and “blue” changes neither the partition nor the resource structure. What matters is which vertices share a conflict-free class and whether every conflict edge has separated endpoints.
<!-- bilingual-en:end -->

颜色只是组标签。交换“红色”和“蓝色”的名字不会改变分组，也不会产生新的资源结构。真正有意义的是：哪些顶点被放进同一个无冲突组，以及每条冲突边的两端是否被分开。

> [!example] A palette need not be fully used / 调色板不必全部用满
> <!-- bilingual-en:start -->
> For the chain $a-b-c$, $c(a)=c(c)=1$ and $c(b)=2$ is proper even if labels $1,2,3,4$ are available. The nonempty classes are $\{a,c\}$ and $\{b\}$. Giving all three vertices label 1 fails on both edges.
> <!-- bilingual-en:end -->
>
> 对链 $a-b-c$，即使可选标签为 $1,2,3,4$，$c(a)=c(c)=1$、$c(b)=2$ 仍合法。非空颜色类只有 $\{a,c\}$ 与 $\{b\}$。若三个顶点全标1，两条边都会违反条件。

<!-- bilingual-en:start -->
## Why coloring represents resource conflicts
<!-- bilingual-en:end -->

## 为什么它能表达资源冲突

<!-- bilingual-en:start -->
In exam scheduling, vertices are exams, an edge joins two exams sharing a student, and colors are time slots. A proper coloring prevents every represented conflict. The same structure models overlapping flights assigned to gates, interfering stations assigned frequencies, or variables with overlapping live ranges assigned registers.
<!-- bilingual-en:end -->

考试排期中，顶点代表考试；两门考试有共同学生时连边；颜色代表时段。proper coloring 保证任何共同学生都不会同时参加两场考试。相同结构也可表达重叠航班分配登机口、相互干扰的电台分配频率，或 live ranges 重叠的变量分配寄存器。

<!-- bilingual-en:start -->
This interpretation requires every edge to mean that its endpoints cannot share one resource class. If edges encode feasible pairings, the problem belongs to [[二分图配对关系|the allowed-edge model for matching]]. Capacity, duration, and preference constraints are likewise not represented automatically by ordinary vertex coloring.
<!-- bilingual-en:end -->

但这个解释成立的前提是：一条边始终表示“这两个对象不能共享同一类资源”。如果边表示允许配对，问题属于 [[二分图配对关系|matching 的允许关系模型]]；如果还存在容量、时长或偏好，普通顶点着色也不会自动表达那些约束。

<!-- bilingual-en:start -->
## Boundaries of the convention
<!-- bilingual-en:end -->

*约定的边界*

<!-- bilingual-en:start -->
Parallel undirected edges repeat the same inequality and add no new coloring constraint. A self-loop would require $c(v)\ne c(v)$, so a graph with a loop has no proper coloring under this rule. Isolated vertices still receive labels but can share an existing label freely. With directed data, state the intended constraint: requiring unequal labels on every directed edge gives the coloring problem for the underlying undirected graph; it does not encode precedence or timing by itself.
<!-- bilingual-en:end -->

无向平行边只会重复同一条不等约束，不增加新的着色限制。自环却会要求 $c(v)\ne c(v)$，因此含自环的图按这条规则不存在合法着色。孤立顶点仍要获得标签，但可以自由复用已有标签。对有向数据，必须先说明约束含义：若每条有向边都要求两端异色，得到的是忽略方向后的无向着色问题；它不会自动表达先后顺序或时刻安排。

<!-- bilingual-en:start -->
After constructing one proper coloring, [[精确色数的上下界|proving optimality]] needs a separate lower bound that rules out every coloring with fewer colors.
<!-- bilingual-en:end -->

构造出一种合法着色后，若要[[精确色数的上下界|证明它已最优]]，还需要另给下界，排除所有颜色更少的着色。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=523|MIT Mathematics for Computer Science]], §12.6.1–12.6.3, PDF523–526: exam conflicts, proper/valid coloring, at-most-$k$ meaning and resource examples.
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Coloring.pdf#page=2|MIT Coloring slides]], PDF2–4, slide labels5–15: conflict graphs, feasible allocations and minimum-color objective.
- Diestel, [Graph Theory, Chapter 5](https://www.math.uni-hamburg.de/home/diestel/books/graph.theory/preview/Ch5.pdf), PDF1 (printed123): coloring as a map and independent color classes. Here unused classes are discarded explicitly; loop/parallel and empty-function boundaries follow directly from the stated rule.
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=523|MIT Mathematics for Computer Science]]，§12.6.1–12.6.3、PDF523–526：考试冲突、合法着色、至多 $k$ 色的含义及资源例子。
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Coloring.pdf#page=2|MIT Coloring slides]]，PDF2–4、幻灯片标签5–15：冲突图、可行资源分配与最少颜色目标。
- Diestel，[Graph Theory 第5章](https://www.math.uni-hamburg.de/home/diestel/books/graph.theory/preview/Ch5.pdf)，PDF1（印123）：函数式着色及独立颜色类。本卡明确去掉未使用的空类；自环、平行边与空函数边界按给定规则直接核验。

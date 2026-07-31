---
aliases:
  - "Bipartite Matching and Hall's Theorem"
  - "Bipartite Graph Matching"
  - "Hall's Marriage Theorem"
  - "二分图匹配"
  - "Hall 定理"
status: source-checked
---

# 二分图匹配与 Hall 定理
<!-- bilingual-en:start -->
*Bipartite matching and Hall's theorem*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 当左侧每个对象只能分配给右侧某些允许对象，而且右侧对象不能被重复占用时，怎样判断能否完成全覆盖分配，并实际找到它？
> **具体锚点：** 学生申请项目：左侧是学生，右侧是项目，边表示该学生符合该项目要求。matching 选出互不共享端点的边；Hall condition 检查任意一组学生是否至少共同拥有同样多的可选项目。
> **核心难点：** 每个人“至少有一个选择”远远不够；真正的障碍是多人挤在过小的共同邻居集合里。augmenting path 把局部改配串成一条交替路径，使 matching 在不制造冲突的前提下增大一条边。
> **为什么重要：** assignment、resource allocation、task–worker compatibility 和许多 scheduling feasibility 问题都可化为 bipartite matching；Hall 定理把“存在匹配”精确等价为“没有 bottleneck subset”。
> **继续：** 先看 [[#从允许关系到 matching]]，再用 [[#Augmenting path：不是贪心定死，而是允许改配]] 理解算法，最后进入 [[#Hall 定理：全覆盖恰好等价于没有瓶颈子集]]；若问题还包含双方偏好与 blocking pair，应改看 stable matching，而不是套 Hall 定理。
> <!-- bilingual-en:start -->
> **What it solves:** When every object on the left may be assigned only to certain allowed objects on the right, and no right-hand object may be used twice, how can we decide whether a full assignment exists and construct one?
> **Concrete anchor:** In student-to-project assignment, students form the left side, projects the right side, and an edge means that a student is eligible for a project. A matching selects edges with no shared endpoint. Hall's condition asks whether every group of students collectively has at least as many eligible projects as students.
> **Central difficulty:** It is nowhere near enough for each individual to have at least one option. The real obstruction is a group crowded into a smaller common pool of options. An augmenting path coordinates a chain of reassignments so that the matching gains one edge without creating a conflict.
> **Why it matters:** Assignments, resource allocation, task–worker compatibility, and many scheduling-feasibility problems reduce to bipartite matching. Hall's theorem makes the existence of a covering matching exactly equivalent to the absence of a bottleneck subset.
> **Continue with:** Begin with [[#从允许关系到 matching|turning allowed relationships into a matching]], use [[#Augmenting path：不是贪心定死，而是允许改配|augmenting paths]] to understand constructive reassignment, and then study [[#Hall 定理：全覆盖恰好等价于没有瓶颈子集|Hall's theorem]]. If the problem includes preferences and blocking pairs, use stable matching rather than Hall's theorem.
> <!-- bilingual-en:end -->

## 从允许关系到 matching
<!-- bilingual-en:start -->
*From allowed relationships to a matching*
<!-- bilingual-en:end -->

一个 finite bipartite graph 写成 $G=(L\cup R,E)$，其中 $L\cap R=\varnothing$，每条边都一端在 $L$、另一端在 $R$。边只表示“这对组合允许”，不自动表示偏好、收益或方向。若还不知道一张图是否 bipartite，可用 BFS/DFS 按奇偶层二着色；失败时会找到 odd cycle，详见 [[图着色与色数#Bipartite、二着色与 odd cycle 是同一件事]]。
<!-- bilingual-en:start -->
A finite bipartite graph is written $G=(L\cup R,E)$, with $L\cap R=\varnothing$ and every edge joining one vertex in $L$ to one in $R$. An edge means only that a pairing is allowed; it does not by itself encode preference, value, or direction. If bipartiteness is not already known, use BFS or DFS to color parity layers with two colors. Failure exposes an odd cycle; see [[图着色与色数#Bipartite、二着色与 odd cycle 是同一件事|bipartiteness, two-colorability, and odd cycles]].
<!-- bilingual-en:end -->

matching 是边集 $M\subseteq E$，其中任意两条边都不共享端点。被某条 $M$ 中的边接触的顶点称为 covered 或 saturated。若 matching 覆盖 $L$ 的每个顶点，它就给出一个 injection $f:L\to R$；只有在 $|L|=|R|$ 时，覆盖 $L$ 也同时覆盖 $R$，成为 perfect matching。
<!-- bilingual-en:start -->
A matching is an edge set $M\subseteq E$ in which no two edges share an endpoint. A vertex incident to an edge of $M$ is covered or saturated. A matching covering every vertex of $L$ defines an injection $f:L\to R$. Only when $|L|=|R|$ does covering $L$ also cover $R$, producing a perfect matching.
<!-- bilingual-en:end -->

必须区分 maximal 与 maximum。maximal matching 只是“当前无法直接再加一条边”，仍可能通过撤下一些已选边并重新分配而变大；maximum matching 才是所有 matchings 中边数最多的。perfect 描述覆盖范围，maximum 描述相对大小，它们也不是同一个词。
<!-- bilingual-en:start -->
Maximal and maximum must be distinguished. A maximal matching merely admits no additional edge directly; it may still become larger after removing and rearranging some selected edges. A maximum matching has the greatest number of edges among all matchings. Perfect describes coverage, whereas maximum describes relative size, so these terms are not interchangeable either.
<!-- bilingual-en:end -->

## Augmenting path：不是贪心定死，而是允许改配
<!-- bilingual-en:start -->
*Augmenting paths: assignments may be rerouted rather than frozen greedily*
<!-- bilingual-en:end -->

相对于 matching $M$，alternating path 的边在“不属于 $M$”与“属于 $M$”之间交替。若它的两个端点都未被 $M$ 覆盖，便是 augmenting path。把路径上的边做 symmetric difference——删掉原 matching edges，加入原 non-matching edges——中间顶点仍各接一条选中边，而两个端点从未匹配变为匹配，所以 matching 大小净增 $1$。
<!-- bilingual-en:start -->
Relative to a matching $M$, an alternating path alternates between edges outside and inside $M$. If both endpoints are uncovered by $M$, it is an augmenting path. Taking the symmetric difference along the path—removing the old matching edges and adding the formerly unmatched edges—leaves every internal vertex incident to one selected edge while matching both endpoints, so the matching grows by exactly one.
<!-- bilingual-en:end -->

例如已有 $a-1$，但新学生 $b$ 只能选项目 $1$，而 $a$ 也可选项目 $2$。路径 $b-1-a-2$ 依次是未选、已选、未选边；翻转后得到 $b-1,a-2$。直接贪心会说“1 已占用”，augmenting path 则发现占用者可以被连锁搬走。
<!-- bilingual-en:start -->
For example, suppose $a-1$ is already selected, a new student $b$ can use only project $1$, and $a$ can also use project $2$. The path $b-1-a-2$ alternates unmatched, matched, unmatched. Flipping it produces $b-1,a-2$. A direct greedy step says that project $1$ is occupied; the augmenting path discovers that its current occupant can be rerouted through a chain of reassignments.
<!-- bilingual-en:end -->

每找到一条 augmenting path 就能把 matching 增大；反过来，若另一个 matching 更大，把两者做 symmetric difference，会分解成 alternating cycles 和 alternating paths，其中必有一条对当前 matching 多一条边的 augmenting path。因此，matching 是 maximum 当且仅当不存在 augmenting path。算法实现可以反复从未匹配左点做 alternating search；朴素重复搜索已足以说明机制，更快实现则会同时分层寻找多条增广路。
<!-- bilingual-en:start -->
Every augmenting path increases a matching. Conversely, if another matching is larger, the symmetric difference of the two decomposes into alternating cycles and paths, and at least one path must contain one more edge from the larger matching—an augmenting path for the current one. Thus, a matching is maximum if and only if no augmenting path exists. An implementation can repeatedly run an alternating search from uncovered left vertices. This simple version explains the mechanism; faster algorithms search for several layered augmenting paths at once.
<!-- bilingual-en:end -->

## Hall 定理：全覆盖恰好等价于没有瓶颈子集
<!-- bilingual-en:start -->
*Hall's theorem: full coverage is equivalent to having no bottleneck subset*
<!-- bilingual-en:end -->

对 $S\subseteq L$，记它在右侧的邻居并集为
<!-- bilingual-en:start -->
For $S\subseteq L$, let the union of its right-hand neighbors be
<!-- bilingual-en:end -->

$$
N(S)=\{r\in R:\exists \ell\in S,\ (\ell,r)\in E\}.
$$

Hall's theorem 断言：finite bipartite graph 存在覆盖全部 $L$ 的 matching，当且仅当
<!-- bilingual-en:start -->
Hall's theorem states that a finite bipartite graph has a matching covering all of $L$ if and only if
<!-- bilingual-en:end -->

$$
|N(S)|\ge |S|\qquad\text{for every }S\subseteq L.
$$

若某个 $S$ 满足 $|N(S)|<|S|$，它就是 bottleneck：$|S|$ 个左点只能使用更少的 $|N(S)|$ 个右点，而 matching 不许共用右点，pigeonhole principle 立即排除全覆盖。因此条件的必要性很直观；关键是它竟也充分——只要没有任何这样的集体拥堵，就一定能通过重排找到全覆盖。
<!-- bilingual-en:start -->
If some $S$ satisfies $|N(S)|<|S|$, it is a bottleneck: the $|S|$ left vertices can use only the smaller pool of $|N(S)|$ right vertices, while a matching forbids sharing a right vertex. The pigeonhole principle immediately rules out full coverage. Necessity is therefore intuitive; the remarkable part is sufficiency—if no collective crowding of this kind exists, some rearrangement always yields full coverage.
<!-- bilingual-en:end -->

### 为什么 Hall condition 足够：alternating-search 证明
<!-- bilingual-en:start -->
*Why Hall's condition is sufficient: an alternating-search proof*
<!-- bilingual-en:end -->

取一个 maximum matching $M$，反设它仍未覆盖 $L$；令非空集合 $S_0$ 是未匹配左点。从 $S_0$ 出发，沿 non-matching edges 从左到右、再沿 matching edges 从右回左，搜到的左右顶点分别记为 $S$ 与 $T$。
<!-- bilingual-en:start -->
Take a maximum matching $M$ and suppose, for contradiction, that it does not cover $L$. Let the nonempty set $S_0$ consist of uncovered left vertices. Starting from $S_0$, traverse unmatched edges from left to right and matching edges from right to left. Let $S$ and $T$ be the reached left and right vertices.
<!-- bilingual-en:end -->

如果 $T$ 中有未匹配右点，搜索树给出一条从 $S_0$ 到它的 augmenting path，与 $M$ maximum 矛盾。因此 $T$ 中每点都已匹配，而且其 matching partner 恰在 $S\setminus S_0$；一一对应给出 $|T|=|S|-|S_0|<|S|$。另一方面，搜索会沿出 $S$ 的所有 non-matching edges，而 $S\setminus S_0$ 的 matching edge 也落在 $T$，故 $N(S)=T$。于是 $|N(S)|<|S|$，违反 Hall condition。反设不能成立，所以 $M$ 覆盖 $L$。
<!-- bilingual-en:start -->
If $T$ contains an uncovered right vertex, the search tree supplies an augmenting path from $S_0$ to that vertex, contradicting the maximal size of $M$. Hence every vertex of $T$ is matched, and its matching partner lies precisely in $S\setminus S_0$. This bijection gives $|T|=|S|-|S_0|<|S|$. Moreover, the search traverses every unmatched edge leaving $S$, while the matching edge of every vertex in $S\setminus S_0$ also ends in $T$, so $N(S)=T$. Therefore $|N(S)|<|S|$, violating Hall's condition. The supposition fails, and $M$ covers $L$.
<!-- bilingual-en:end -->

> [!source] 定理核验
> MIT 6.042J Section 12.5 给出 bipartite graph、matching、covered vertex、perfect matching、neighborhood、bottleneck 与 Hall condition 的正式定义，并以强归纳证明 Hall condition 充分；本课程 Session 22 用上面的 alternating-search 论证连接 maximum matching、augmenting path 与 bottleneck certificate；MIT 18.997 Lecture 1 核验 augmenting path 的 symmetric-difference 操作与 Berge theorem。
> <!-- bilingual-en:start -->
> MIT 6.042J Section 12.5 gives the formal definitions of bipartite graphs, matchings, covered vertices, perfect matchings, neighborhoods, bottlenecks, and Hall's condition, and proves sufficiency by strong induction. Course Session 22 uses the alternating-search argument above to connect maximum matchings, augmenting paths, and bottleneck certificates. MIT 18.997 Lecture 1 supports symmetric-difference augmentation and Berge's theorem.
> <!-- bilingual-en:end -->

## 完整例子：先找匹配，再找失败证书
<!-- bilingual-en:start -->
*Worked example: first find a matching, then find a certificate of failure*
<!-- bilingual-en:end -->

令 $L=\{a,b,c\}$、$R=\{1,2,3\}$，并设 $N(a)=\{1,2\}$、$N(b)=\{1\}$、$N(c)=\{2,3\}$。可以先配 $b-1$，再配 $a-2$，最后配 $c-3$，得到覆盖 $L$ 的 perfect matching。Hall condition 也成立：最紧的子集如 $\{a,b\}$ 有邻居 $\{1,2\}$，大小恰好相等，但没有子集拥有更少的邻居。
<!-- bilingual-en:start -->
Let $L=\{a,b,c\}$ and $R=\{1,2,3\}$, with $N(a)=\{1,2\}$, $N(b)=\{1\}$, and $N(c)=\{2,3\}$. Match $b-1$, then $a-2$, and finally $c-3$ to obtain a perfect matching covering $L$. Hall's condition also holds. A tight subset such as $\{a,b\}$ has neighborhood $\{1,2\}$ of equal size, but no subset has fewer neighbors than vertices.
<!-- bilingual-en:end -->

现在删除 $a-2$，使 $a$ 与 $b$ 都只能连接项目 $1$。单看每个学生，degree 都至少为 1；但取 $S=\{a,b\}$，有 $N(S)=\{1\}$，所以 $|N(S)|=1<2=|S|$。这一个集合就是无法覆盖的完整证书：无论怎样重排，$a,b$ 中最多一个人得到项目。
<!-- bilingual-en:start -->
Now delete edge $a-2$, so both $a$ and $b$ can use only project $1$. Individually, every student still has degree at least one. But for $S=\{a,b\}$, $N(S)=\{1\}$ and hence $|N(S)|=1<2=|S|$. This single set is a complete certificate of infeasibility: regardless of rearrangement, at most one of $a$ and $b$ can receive a project.
<!-- bilingual-en:end -->

## 建模与诊断边界
<!-- bilingual-en:start -->
*Modeling and diagnostic boundaries*
<!-- bilingual-en:end -->

| 容易混淆的判断 | 为什么不够或不对 | 应该检查什么 |
|---|---|---|
| “每个左点 degree 至少 1，所以能全覆盖” | 多个左点可能只有同一个邻居 | 检查所有 relevant subsets 的 $|N(S)|\ge|S|$，或运行 matching algorithm 并取得 bottleneck certificate |
| “matching 已 maximal，所以是 maximum” | 可能需要先撤边再沿 alternating path 重配 | 搜索 augmenting path |
| “覆盖 $L$ 就是 perfect” | 当 $|R|>|L|$ 时仍会留下右点 | 分清 covers $L$ 与 covers $L\cup R$ |
| “没有 perfect matching，所以实例无解” | 目标可能只要求最大覆盖或加权最优 | 先定义 coverage、cardinality 与 weight 目标 |
| “Hall 定理处理双方偏好” | Hall 只使用 allowed edges | 有 preferences 与 blocking pairs 时使用 stable matching 模型 |
<!-- bilingual-en:start -->
| Tempting claim | Why it is insufficient or wrong | What to check instead |
|---|---|---|
| “Every left vertex has degree at least one, so full coverage exists.” | Several left vertices may share the same sole neighbor | Check $|N(S)|\ge|S|$ for relevant subsets, or run a matching algorithm that returns a bottleneck certificate |
| “The matching is maximal, so it is maximum.” | Increasing it may require removing edges and rerouting along an alternating path | Search for an augmenting path |
| “Covering $L$ means the matching is perfect.” | Right vertices remain uncovered when $|R|>|L|$ | Distinguish covering $L$ from covering all of $L\cup R$ |
| “No perfect matching means the instance has no solution.” | The objective may require only maximum coverage or maximum weight | Define the coverage, cardinality, and weight objective first |
| “Hall's theorem handles bilateral preferences.” | Hall uses only allowed edges | Use a stable-matching model when preferences and blocking pairs matter |
<!-- bilingual-en:end -->

实际建模时先固定五件事：两侧分别是什么；一条边代表什么资格；是否必须覆盖左侧全部；右侧是否有 capacity；目标只问 feasibility，还是还要最大 cardinality、最小成本或最大权重。有容量的右点可以在整数容量下复制为多个 slots，或改写为 flow；带权目标则不由 Hall 条件单独决定。
<!-- bilingual-en:start -->
In an application, first fix five choices: what each side represents, what eligibility an edge means, whether every left vertex must be covered, whether right vertices have capacities, and whether the objective is mere feasibility, maximum cardinality, minimum cost, or maximum weight. Integer capacities can be modeled by cloning a right vertex into slots or by using flow. Hall's condition alone does not determine a weighted optimum.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 用自己的话解释：为什么只检查每个左点都有邻居不够？
<!-- bilingual-en:start -->
*Explain in your own words: why is checking that every left vertex has a neighbor insufficient?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 因为匹配禁止共用右点。每个左点可以各自有选择，却全部集中在同一个很小的邻居集合中；Hall condition 正是检查这种集体拥堵。
> <!-- bilingual-en:start -->
> A matching forbids sharing a right vertex. Every left vertex may have an option while all options are concentrated in one small common neighborhood. Hall's condition detects precisely this collective crowding.
> <!-- bilingual-en:end -->

### 路径 $b-1-a-2$ 中，若 $a-1$ 已匹配而两端 $b,2$ 未匹配，翻转后发生什么？
<!-- bilingual-en:start -->
*On path $b-1-a-2$, if $a-1$ is matched and endpoints $b$ and $2$ are unmatched, what happens after the flip?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 删除 $a-1$，加入 $b-1$ 与 $a-2$。原 matching 增加一条边，且每个顶点仍最多接一条 matching edge。
> <!-- bilingual-en:start -->
> Remove $a-1$ and add $b-1$ and $a-2$. The matching gains one edge, and every vertex remains incident to at most one matching edge.
> <!-- bilingual-en:end -->

### 已知 $S\subseteq L$ 有 7 个点而 $N(S)$ 只有 6 个点，能否覆盖 $L$？
<!-- bilingual-en:start -->
*If some $S\subseteq L$ has seven vertices but $N(S)$ has only six, can a matching cover $L$?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不能。$S$ 的七个左点必须使用六个右点且不能重复，pigeonhole principle 直接给出矛盾；$S$ 是 bottleneck certificate。
> <!-- bilingual-en:start -->
> No. The seven left vertices would have to use six distinct right vertices without repetition, contradicting the pigeonhole principle. The set $S$ is a bottleneck certificate.
> <!-- bilingual-en:end -->

### 一个 matching 没有 augmenting path，能得出什么结论？
<!-- bilingual-en:start -->
*What follows if a matching has no augmenting path?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它是 maximum matching。若存在更大的 matching，两者的 symmetric difference 中必出现一条对当前 matching 多一条边的 alternating path，也就是 augmenting path。
> <!-- bilingual-en:start -->
> It is a maximum matching. If a larger matching existed, the symmetric difference would contain an alternating path with one more edge from the larger matching, which would be an augmenting path for the current one.
> <!-- bilingual-en:end -->

### 一家企业既有“岗位是否允许申请”又有双方偏好，应先区分哪两个问题？
<!-- bilingual-en:start -->
*If a firm has both job eligibility constraints and preferences on both sides, which two questions should be separated first?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 先问 allowed-edge graph 中是否存在所需覆盖的 matching；再问带偏好的 matching 是否没有 blocking pair。前者是 bipartite matching/Hall，后者是 stable matching，目标并不相同。
> <!-- bilingual-en:start -->
> First ask whether the allowed-edge graph contains a matching with the required coverage. Then ask whether a preference-based matching has no blocking pair. The former is bipartite matching and Hall's theorem; the latter is stable matching, with a different objective.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf]] Section 12.5：支持 bipartite graph、matching、covered/perfect、neighborhood、bottleneck 与 Hall theorem 的正式定义，以及 Hall condition 充分性的证明。
<!-- bilingual-en:start -->
- Section 12.5 of [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] supports the formal definitions of bipartite graphs, matchings, covered and perfect matchings, neighborhoods, bottlenecks, and Hall's theorem, together with the proof that Hall's condition is sufficient.
<!-- bilingual-en:end -->
- [[01_Math/07-Mathematics for Computer Science/02_Structures.md#Session 22 — Stable Matching and Hall's Theorem]]：支持本课程使用的符号、Hall condition 的 alternating-search 证明、stable matching 与 graph matching 的区分，以及课程中的 bottleneck 诊断。
<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/02_Structures.md#Session 22 — Stable Matching and Hall's Theorem|Course Session 22 on stable matching and Hall's theorem]] supports the course notation, the alternating-search proof of Hall's condition, the distinction between stable and graph matching, and the course's bottleneck diagnostics.
<!-- bilingual-en:end -->
- MIT OpenCourseWare 18.997, [Lecture 1: Non-Bipartite Matching](https://ocw.mit.edu/courses/18-997-topics-in-combinatorial-optimization-spring-2004/cb34901551f24affa0c147af1cb9151a_co_lec1.pdf)：核验 alternating/augmenting path、沿路径取 symmetric difference 后 matching 增大 1，以及 Berge theorem 的 maximum-matching 判据。
<!-- bilingual-en:start -->
- MIT OpenCourseWare 18.997, [Lecture 1: Non-Bipartite Matching](https://ocw.mit.edu/courses/18-997-topics-in-combinatorial-optimization-spring-2004/cb34901551f24affa0c147af1cb9151a_co_lec1.pdf), supports alternating and augmenting paths, the one-edge increase after taking a symmetric difference along the path, and Berge's maximum-matching criterion.
<!-- bilingual-en:end -->
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_bip_mtchig.pdf]] 与 [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_halls_thorem.pdf]]：核对 bipartite matching 例子、matching condition 与 Hall 定理的课程表述。
<!-- bilingual-en:start -->
- The local MIT 6.042J slides on [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_bip_mtchig.pdf|bipartite matching]] and [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_halls_thorem.pdf|Hall's theorem]] support the course examples, matching condition, and theorem statement.
<!-- bilingual-en:end -->

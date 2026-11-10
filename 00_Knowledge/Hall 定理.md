---
aliases:
  - Hall 定理
  - Hall's Marriage Theorem
  - Hall condition
student_os: knowledge-atom
atom_id: MCS-MATCH-008
atom_type: theorem
status: source-checked
requires:
  - "[[二分图]]"
  - "[[左侧覆盖与完美匹配]]"
  - "[[匹配定义]]"
  - "[[增广路对称差]]"
part_of:
  - "[[二分图匹配与 Hall 定理.canvas]]"
---

<!-- bilingual-en:start -->
# Hall's marriage theorem
<!-- bilingual-en:end -->
*Hall 定理*

<!-- bilingual-en:start -->
Suppose each student has a list of acceptable projects, and each project can take at most one student. Giving every student at least one option is not enough: several students may be competing for too few projects. **Hall’s theorem** says that this is the only obstruction. Every group of students must collectively have at least as many available projects as students in that group.
<!-- bilingual-en:end -->
假设每名学生都有一份允许选择的项目清单，每个项目至多接收一名学生。每名学生都至少有一个选项，还不够：若干学生可能共同挤在过少的项目上。**Hall 定理**说明，这种集体拥挤就是唯一障碍：每组学生合起来能选择的项目数，必须至少等于这组学生的人数。

<!-- bilingual-en:start -->
## Count distinct neighbors of a whole group
<!-- bilingual-en:end -->
*数一整组对象的不同邻居*

<!-- bilingual-en:start -->
Work in a finite simple undirected [[二分图|bipartite graph]] $G=(L\cup R,E)$, with disjoint sides $L$ and $R$. The designated side $L$ is the group we require to cover. A [[匹配定义|matching]] selects edges with no repeated endpoint; [[左侧覆盖与完美匹配|covering the left side]] means every left vertex occurs in one selected edge. For $S\subseteq L$, its **neighborhood** $N(S)$ is the set of all right vertices adjacent to at least one vertex of $S$. Take the union of their options, counting each right vertex once, not once per incident edge.
<!-- bilingual-en:end -->
在有限简单无向[[二分图]] $G=(L\cup R,E)$ 中讨论，左右两侧 $L,R$ 不相交。指定的 $L$ 是必须覆盖的一组对象。[[匹配定义|匹配]]的边不重复使用任何端点；[[左侧覆盖与完美匹配|覆盖左侧]]表示每个左点都出现在一条选中边中。对 $S\subseteq L$，其**邻居集** $N(S)$ 是至少与一个 $S$ 中顶点相邻的全部右点。这里合并各点的选项，同一个右点只数一次，不按连接到它的边数重复计算。

> [!important] Hall’s condition / Hall 条件
> <!-- bilingual-en:start -->
> Let $G=(L\cup R,E)$ be a finite bipartite graph. For $S\subseteq L$, define its neighbourhood by
> $$
> N(S)=\{r\in R:\text{some }\ell\in S\text{ satisfies }(\ell,r)\in E\}.
> $$
> A matching covers all of $L$ if and only if
> $$
> |N(S)|\ge |S|\qquad\text{for every }S\subseteq L.
> $$
> <!-- bilingual-en:end -->
> 设 $G=(L\cup R,E)$ 是 finite bipartite graph。对 $S\subseteq L$，定义
> $$
> N(S)=\{r\in R:\text{存在 }\ell\in S\text{ 使 }(\ell,r)\in E\}.
> $$
> 存在覆盖全部 $L$ 的 matching，当且仅当
> $$
> |N(S)|\ge |S|\qquad\text{对每个 }S\subseteq L.
> $$

<!-- bilingual-en:start -->
Here $(\ell,r)\in E$ records the allowed undirected pair using its left and right roles; it does not make the graph directed. Vertical bars denote the number of elements of a finite set. The quantifier includes $S=\varnothing$, for which $N(S)=\varnothing$ and $0\ge0$ holds.
<!-- bilingual-en:end -->
这里 $(\ell,r)\in E$ 按左右角色记录允许的无向配对，不表示图被改成有向图。竖线表示有限集合的元素个数。量词也包括 $S=\varnothing$：此时 $N(S)=\varnothing$，条件是成立的 $0\ge0$。

<!-- bilingual-en:start -->
For $L=\{a,b\}$, $R=\{1,2\}$ and edges $a1,b1$, each singleton has one neighbor, but $N(\{a,b\})=\{1\}$ has size one, smaller than two. Left coverage is impossible. Conversely, with edges $a1,a2,b2$, the four subsets $\varnothing,\{a\},\{b\},\{a,b\}$ have neighborhood sizes $0,2,1,2$. All four inequalities hold, and $\{a1,b2\}$ covers $L$.
<!-- bilingual-en:end -->
取 $L=\{a,b\}$、$R=\{1,2\}$，只允许边 $a1,b1$。每个单点都有一个邻居，但 $N(\{a,b\})=\{1\}$ 只有一个元素，小于两名学生的人数，所以无法覆盖左侧。相反，若允许边为 $a1,a2,b2$，四个子集 $\varnothing,\{a\},\{b\},\{a,b\}$ 的邻居数分别为 $0,2,1,2$，全部不等式成立，而 $\{a1,b2\}$ 就覆盖了 $L$。

<!-- bilingual-en:start -->
## Necessity: partners must be different
<!-- bilingual-en:end -->
*必要性：伙伴必须各不相同*

<!-- bilingual-en:start -->
Necessity comes from injectivity: if a matching covers $L$, every vertex of $S$ is assigned to a distinct vertex in $N(S)$, so $|S|\le |N(S)|$. Sufficiency is deeper: if no subset of the left side is crowded into a smaller common neighbourhood, some global rearrangement covers $L$.
<!-- bilingual-en:end -->
必要性来自单射：若 matching 覆盖 $L$，那么 $S$ 中每个左点都匹配到 $N(S)$ 中一个不同的右点，因此 $|S|\le |N(S)|$。充分性更深：只要任何左侧子集都没有挤进更小的共同邻居集，就一定存在某种全局重排覆盖 $L$。

<!-- bilingual-en:start -->
## Sufficiency: an unmatched root would expose a shortage
<!-- bilingual-en:end -->
*充分性：未匹配起点会暴露一个缺额集合*

<!-- bilingual-en:start -->
Assume Hall’s inequalities hold for every $S\subseteq L$. Because the graph is finite, it has finitely many matchings and hence one, $M$, of largest cardinality. If it covers $L$, we are done. Otherwise choose an unmatched left vertex $u$. We will derive a set violating Hall’s inequality.
<!-- bilingual-en:end -->
假设每个 $S\subseteq L$ 都满足 Hall 不等式。图有限，所以匹配的候选集合有限，其中存在边数最大的一个，记为 $M$。若它覆盖 $L$，结论已经成立；否则取一个未匹配左点 $u$，下面将构造一个违反 Hall 不等式的集合。

<!-- bilingual-en:start -->
Temporarily orient each unselected edge from $L$ to $R$ and each selected edge from $R$ to $L$. This is only a device for describing allowed reassignment routes; the original graph remains undirected. Let $S$ be the left vertices reachable from $u$ along these arrows, including $u$ itself, and let $T$ be the reachable right vertices. A reachable vertex has a shortest arrow-respecting route, which repeats no vertex, since a repeated portion could be deleted. Such a route alternates between unselected and selected edges.
<!-- bilingual-en:end -->
暂时把未选边定向为从 $L$ 到 $R$，把已选边定向为从 $R$ 到 $L$。这只是描述允许重配路线的工具，原图仍是无向图。令 $S$ 为从 $u$ 沿这些箭头可达的左点集合，包括 $u$ 本身；令 $T$ 为可达的右点集合。任何可达点都有一条最短的顺箭头路线，它不重复顶点，否则可删掉重复部分。这条路线在未选与已选边之间交替。

<!-- bilingual-en:start -->
Every vertex of $T$ must be matched. If a right vertex $r\in T$ were unmatched, the simple route from the unmatched $u$ to $r$ would be an augmenting path. [[增广路对称差|Flipping it]] would produce a matching larger than $M$, contradicting our choice. Thus each $r\in T$ has a unique matched partner in $S\setminus\{u\}$, reached by its selected arrow. Conversely, every left vertex in $S\setminus\{u\}$ is reached on a last arrow from a right vertex, and such an arrow must be its unique selected edge. Those right vertices belong to $T$. Matching partners therefore give a bijection:
<!-- bilingual-en:end -->
$T$ 中每个顶点都必须已匹配。如果某个右点 $r\in T$ 未匹配，那么从未匹配起点 $u$ 到它的简单路线就是增广路；[[增广路对称差|翻转]]会得到比 $M$ 更大的匹配，与选择矛盾。因此，每个 $r\in T$ 都有唯一的匹配伙伴，沿其已选箭头到达 $S\setminus\{u\}$ 中。反过来，$S\setminus\{u\}$ 中每个左点都由某个右点通过最后一条箭头到达，而这种箭头必是该左点唯一的已选边；那个右点属于 $T$。所以匹配伙伴给出双射：

$$
T\longleftrightarrow S\setminus\{u\},\qquad |T|=|S|-1.
$$

<!-- bilingual-en:start -->
We also have $N(S)=T$. Each reached right vertex has an incoming arrow from a reached left vertex, so $T\subseteq N(S)$. In the other direction, take a neighbor $r$ of $\ell\in S$. If $\ell r\notin M$, the arrow $\ell\to r$ makes $r$ reachable. If $\ell r\in M$, then $\ell\ne u$ and $r$ is the unique matched predecessor already used to reach $\ell$, so it too is reachable. Hence $N(S)\subseteq T$.
<!-- bilingual-en:end -->
还要证明 $N(S)=T$。每个可达右点都有一条来自可达左点的入箭头，所以 $T\subseteq N(S)$。反过来，取 $\ell\in S$ 的任一邻居 $r$：若 $\ell r\notin M$，则箭头 $\ell\to r$ 让 $r$ 可达；若 $\ell r\in M$，则 $\ell\ne u$，而 $r$ 是到达 $\ell$ 时已经经过的唯一匹配前驱，所以也可达。故 $N(S)\subseteq T$。

<!-- bilingual-en:start -->
Together these statements give $|N(S)|=|S|-1<|S|$, contradicting Hall’s condition. The assumed unmatched root cannot exist, so $M$ covers $L$. This completes both directions without assuming a flow theorem or relying on an unproved search algorithm. The separate [[交替搜索的增广与缺额证书|alternating-search procedure]] explains how to organize the same reachability argument computationally.
<!-- bilingual-en:end -->
合起来得到 $|N(S)|=|S|-1<|S|$，违反 Hall 条件。因此假设的未匹配起点不可能存在，$M$ 必覆盖 $L$。至此两方向均已证明，无须预设网络流定理，也没有把未证明的搜索算法当作依据。另见[[交替搜索的增广与缺额证书|交替搜索过程]]，了解怎样在计算中组织同一可达性论证。

<!-- bilingual-en:start -->
## Which subsets and which side?
<!-- bilingual-en:end -->
*哪些子集、指定哪一侧？*

<!-- bilingual-en:start -->
Checking only $S=L$ is also insufficient. For example, let $L=\{a,b,c\}$ and $R=\{1,2,3\}$, with edges $a1,b1,c2,c3$. Then $|N(L)|=3=|L|$, and every left vertex has a neighbor, yet $S=\{a,b\}$ has only the neighbor $1$. A single violating subset is enough to rule out coverage; the universal condition cannot be replaced by just singleton and whole-side checks.
<!-- bilingual-en:end -->
只检查 $S=L$ 也不够。例如 $L=\{a,b,c\}$、$R=\{1,2,3\}$，允许边为 $a1,b1,c2,c3$。虽然 $|N(L)|=3=|L|$，每个左点也都有邻居，但 $S=\{a,b\}$ 只有邻居 $1$。一个违反条件的子集就足以否定覆盖；全部子集的量词不能被单点检查与整侧检查取代。

<!-- bilingual-en:start -->
The theorem must name the side to be covered. To cover $R$, swap the roles and check every subset of $R$. If $|L|=|R|$, covering $L$ automatically gives a perfect matching. Weights, capacities, and preferences do not appear in Hall's condition and therefore require separate modelling.
<!-- bilingual-en:end -->
定理必须指定被覆盖的一侧。若目标是覆盖 $R$，应交换左右角色并对 $R$ 的所有子集检查条件；若 $|L|=|R|$，覆盖 $L$ 自动成为 perfect matching。权重、容量和偏好不会出现在 Hall 条件里，因此需要另行建模。

<!-- bilingual-en:start -->
If $L=\varnothing$, the empty matching covers it and the only subset condition is $0\ge0$. If $R=\varnothing$ but $L\ne\varnothing$, any singleton left subset violates the condition. An isolated left vertex is the same one-point obstruction; an isolated right vertex need not obstruct left coverage. Finite graphs, including these empty-side cases, are the scope of this statement and proof.
<!-- bilingual-en:end -->
若 $L=\varnothing$，空匹配就覆盖它，唯一子集条件为 $0\ge0$。若 $R=\varnothing$ 而 $L\ne\varnothing$，任取一个左侧单点就违反条件。孤立左点是同样的单点障碍；孤立右点却未必阻止覆盖左侧。本定理与证明的范围是有限图，也包括这些空侧情形。

> [!question]- Why not check one vertex at a time? / 回忆与辨别
> <!-- bilingual-en:start -->
> Why is it insufficient that every left vertex has a neighbor? Several left vertices may share a neighborhood smaller than their own number; Hall’s condition checks the entire group.
> <!-- bilingual-en:end -->
> 为什么只检查每个单点 $\{\ell\}$ 都有邻居不够？
>
> **答案：** 多个左点可能各自有邻居，却全部共享同一个过小的邻居集合；Hall 条件必须检查所有左侧子集。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=519|MIT Mathematics for Computer Science, §12.5.2, PDF519–521 / printed511–513]]: the neighborhood counting condition, both directions of Hall’s theorem, and the formal left-covering statement (Theorem12.5.4).
- Diestel, [Graph Theory, Chapter2](https://www.math.uni-hamburg.de/home/diestel/books/graph.theory/preview/Ch2.pdf), PDF2–4 / printed38–40, especially Theorem2.1.2 and its first proof: alternating reachability from an unmatched left root; the argument above expands the partner bijection and equality of neighbor and reached-right sets.
- MIT 18.200, [Lecture11–12: Network Flows and Matching](https://ocw.mit.edu/courses/18-200-principles-of-discrete-applied-mathematics-spring-2024/mit18_200_s24_lec11-12.pdf), PDF8–9, Theorem4: independent cross-check of the designated-side Hall condition through the flow formulation. The proof above does not require that formulation.
<!-- bilingual-en:end -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=519|MIT Mathematics for Computer Science]]，§12.5.2，PDF519–521／印511–513页：邻居计数条件、Hall 双向结论及 Theorem12.5.4 的指定左侧覆盖表述。
- Diestel，[Graph Theory 第2章](https://www.math.uni-hamburg.de/home/diestel/books/graph.theory/preview/Ch2.pdf)，PDF2–4／印38–40页，尤其 Theorem2.1.2 第一种证明：从未匹配左侧起点作交替可达性论证；上文展开伙伴双射与邻居集等于可达右点集的细节。
- MIT 18.200，[Lecture11–12: Network Flows and Matching](https://ocw.mit.edu/courses/18-200-principles-of-discrete-applied-mathematics-spring-2024/mit18_200_s24_lec11-12.pdf)，PDF8–9，Theorem4：通过流模型交叉核验指定一侧的 Hall 条件；上文证明不依赖该模型。

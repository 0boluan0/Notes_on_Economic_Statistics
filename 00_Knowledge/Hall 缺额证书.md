---
aliases:
- Hall 瓶颈
- Hall bottleneck certificate
- Hall-deficient set
- Deficient set
student_os: knowledge-atom
atom_id: MCS-MATCH-009
atom_type: certificate
status: source-checked
requires:
  - "[[Hall 定理]]"
  - "[[二分图]]"
  - "[[匹配定义]]"
  - "[[左侧覆盖与完美匹配]]"
part_of:
  - "[[二分图匹配与 Hall 定理.canvas]]"
---

<!-- bilingual-en:start -->
# Hall deficiency certificates
<!-- bilingual-en:end -->
*Hall 缺额证书*

<!-- bilingual-en:start -->
Two students may each have an acceptable project and still be impossible to place: if both can use only the same one-person project, one must be left out. A **Hall deficiency certificate** identifies such a group whose combined choices are too few. It proves that no rearrangement of the assignments can serve everyone in that group.
<!-- bilingual-en:end -->
两名学生各有一个允许选择的项目，仍可能无法同时分配：如果两人都只能使用同一个单人项目，就必有一人落空。**Hall 缺额证书**找出这样一组对象：他们合起来能选择的伙伴太少，因此无论怎样重排，都不能让组内人人得到配对。

<!-- bilingual-en:start -->
## Count distinct possible partners
<!-- bilingual-en:end -->
*数不同的可选伙伴*

<!-- bilingual-en:start -->
Use a finite simple undirected [[二分图|bipartite graph]] $G=(L\cup R,E)$ with disjoint, designated sides $L$ and $R$. A [[匹配定义|matching]] chooses edges with no shared endpoint. The goal is [[左侧覆盖与完美匹配|to cover]] $L$: every left vertex must be incident to a chosen edge. For a subset $S\subseteq L$, define its **neighborhood** by
<!-- bilingual-en:end -->
考虑有限简单无向[[二分图]] $G=(L\cup R,E)$，其中指定的两侧 $L,R$ 不相交。[[匹配定义|匹配]]选取互不共享端点的边。目标是[[左侧覆盖与完美匹配|覆盖]] $L$，即每个左点都接触一条选中边。对左侧子集 $S\subseteq L$，定义它的**邻居集**为

$$N(S)=\{r\in R:\text{there exists }\ell\in S\text{ with }\{\ell,r\}\in E\}.$$

<!-- bilingual-en:start -->
Thus $N(S)$ contains all right vertices adjacent to at least one member of $S$. The braces $\{\ell,r\}$ describe an undirected edge. Combine the option lists and remove duplicates: a project available to five students is still just one possible partner. The notation $|S|$ means the number of elements of the finite set $S$.
<!-- bilingual-en:end -->
也就是说，$N(S)$ 包含至少与一个 $S$ 中左点相邻的全部右点；$\{\ell,r\}$ 表示无向边。把选项清单合并后要去重：一个项目即使被五名学生共同选择，也仍只有一个可用伙伴。$|S|$ 表示有限集合 $S$ 的元素个数。

> [!important] A certificate of insufficient supply · 伙伴不足的证书
> <!-- bilingual-en:start -->
> If $S\subseteq L$ satisfies $|N(S)|<|S|$, it is a **Hall-deficient set**, or **bottleneck**. Put $d=|S|-|N(S)|>0$. Every matching leaves at least $d$ vertices of $S$ unmatched. In particular, no matching covers $L$.
> <!-- bilingual-en:end -->
> 若 $S\subseteq L$ 满足 $|N(S)|<|S|$，就称它为 **Hall 缺额集合**或**瓶颈**。令 $d=|S|-|N(S)|>0$，则任意匹配都至少留下 $S$ 中的 $d$ 个顶点未匹配，特别地，不存在覆盖 $L$ 的匹配。

<!-- bilingual-en:start -->
## Why one subset proves impossibility
<!-- bilingual-en:end -->
*为什么一个子集就能证明不可能*

<!-- bilingual-en:start -->
Fix any matching $M$. Every covered vertex of $S$ has its partner in $N(S)$, and different covered vertices have different partners, since a matching cannot reuse a right endpoint. Hence at most $|N(S)|$ members of $S$ are covered. Subtracting from $|S|$ leaves at least $d$ uncovered. This is the pigeonhole argument: distinct assignments require distinct available partners. It applies to every matching, not merely the current attempt.
<!-- bilingual-en:end -->
任取匹配 $M$。$S$ 中每个已覆盖顶点的伙伴都属于 $N(S)$；不同已覆盖顶点的伙伴也必须不同，否则重复使用了右端点。因此 $S$ 中至多有 $|N(S)|$ 个点被覆盖，从 $|S|$ 中减去这个上限，至少剩下 $d$ 个未覆盖点。这就是鸽巢计数：互不冲突的分配需要不同的可选伙伴。它约束所有匹配，而不只是当前试出的那一个。

<!-- bilingual-en:start -->
The same count gives a bound on the total number of selected edges. At most $|L|-|S|$ additional left vertices lie outside $S$, and each matching edge covers exactly one left vertex, so
<!-- bilingual-en:end -->
同一计数还给出匹配总边数的上界。$S$ 外至多还有 $|L|-|S|$ 个左点可覆盖，每条匹配边恰覆盖一个左点，所以

$$|M|\le |N(S)|+(|L|-|S|)=|L|-d.$$

<!-- bilingual-en:start -->
For an arbitrary deficient set, this is an upper bound; the graph may contain further obstructions and have an even smaller maximum matching. No enumeration of possible matchings is needed to verify the bound.
<!-- bilingual-en:end -->
对任意给定缺额集合，这只是上界；图中可能还有别的障碍，使最大匹配更小。验证这个上界不需要枚举所有可能匹配。

> [!example] Enough projects overall, but too few for a group · 总数够，局部仍不足
> <!-- bilingual-en:start -->
> Let $L=\{a,b,c\}$, $R=\{1,2,3\}$, with exactly the edges $a1,b1,c2,c3$; $a1$ abbreviates $\{a,1\}$. Every student has an option, the two sides have equal sizes, and $N(L)=R$. Nevertheless, for $S=\{a,b\}$, $N(S)=\{1\}$ and $d=2-1=1$. Any matching has at most two edges. The matching $\{a1,c2\}$ attains that bound, but student $b$ cannot also receive project $1$.
> <!-- bilingual-en:end -->
> 设 $L=\{a,b,c\}$、$R=\{1,2,3\}$，恰有边 $a1,b1,c2,c3$；$a1$ 简记 $\{a,1\}$。每名学生都有选项，两侧等大，而且 $N(L)=R$。但取 $S=\{a,b\}$，有 $N(S)=\{1\}$、$d=2-1=1$，所以任何匹配至多有两条边。$\{a1,c2\}$ 达到此上界，却不能再把项目 $1$ 分给学生 $b$。

<!-- bilingual-en:start -->
## Verify the set, not just a failed attempt
<!-- bilingual-en:end -->
*验证集合，而不只是看到一次失败*

<!-- bilingual-en:start -->
To check a proposed certificate, verify $S\subseteq L$, form the complete union of the neighbors of its members, and compare the two sizes. A partial list of inspected neighbors can undercount $N(S)$ and create a false shortage. Checking only individual degrees or only $S=L$ is likewise insufficient, as the example shows.

The empty set is never deficient: $N(\varnothing)=\varnothing$ gives $0=0$. An isolated left vertex supplies a one-point certificate. If $L=\varnothing$, there is no deficient subset and the empty matching covers $L$. If $R=\varnothing$ and $L\ne\varnothing$, every nonempty left subset is deficient. These are finite-set statements; subtraction of infinite cardinalities is not being used.
<!-- bilingual-en:end -->
核验别人给出的证书时，应确认 $S\subseteq L$，取出组内所有点的完整邻居并集，再比较两个大小。只看部分已检查的邻居，会低估 $N(S)$，制造假缺额。只检查每个点的度数，或只检查整个 $L$，也不够，前例已经说明。

空集永远不缺额：$N(\varnothing)=\varnothing$ 给出 $0=0$。孤立左点给出单点证书。若 $L=\varnothing$，没有缺额子集，空匹配覆盖 $L$；若 $R=\varnothing$ 而 $L\ne\varnothing$，任何非空左侧子集都缺额。这里讨论有限集合，不使用无限基数相减。

<!-- bilingual-en:start -->
Finding one deficient set proves infeasibility immediately. Failing to find one in a few guesses proves nothing. [[Hall 定理|Hall's theorem]] states the converse with its full quantifier: if no subset of $L$ is deficient, a matching covering $L$ exists. [[交替搜索的增广与缺额证书|Alternating search]] explains how an exhaustive unsuccessful search from unmatched left vertices produces a certificate, instead of guessing subsets. In particular, a maximum matching that does not cover $L$ leads to such a set.
<!-- bilingual-en:end -->
发现一个缺额集合就能立即证明不可行；随便试几个子集没有发现缺额，却不能证明可行。[[Hall 定理]]给出带完整量词的反方向：若 $L$ 的所有子集都不缺额，就存在覆盖 $L$ 的匹配。[[交替搜索的增广与缺额证书|交替搜索]]说明怎样从未匹配左点出发，在穷尽搜索而无法增广时构造证书，无需猜子集。特别地，未覆盖 $L$ 的最大匹配会导出这种集合。

> [!question]- Seven vertices, six neighbors · 七个顶点，六个邻居
> <!-- bilingual-en:start -->
> If $|S|=7$ and $|N(S)|=6$, what is proved? Every matching leaves at least one member of $S$ uncovered, so full left coverage is impossible. The counts alone do not promise that six members of $S$ can actually be covered.
> <!-- bilingual-en:end -->
> 若 $|S|=7$、$|N(S)|=6$，能证明什么？任意匹配至少遗漏 $S$ 中一人，所以不可能覆盖全部左侧；仅凭这两个数量，不能保证确实能够覆盖其中六人。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=519|MIT Mathematics for Computer Science, §12.5.2, PDF519–521 / printed511–513]]: distinct-neighbor counting, Definition12.5.3's bottleneck, and Theorem12.5.4's left-coverage criterion. The displayed deficit bound is the same partner-counting argument with the uncovered vertices counted explicitly.
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_halls_thorem.pdf#page=1|MIT 6.042J Hall's Theorem slides, PDF1–2 / slides Hall.2–Hall.7]]: the designated-side injective assignment and the all-subsets condition. The slides use $E(S)$ for the neighbor set that is denoted $N(S)$ here.
<!-- bilingual-en:end -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=519|MIT Mathematics for Computer Science]]，§12.5.2，PDF519–521／印511–513页：不同邻居的计数、Definition12.5.3 的瓶颈定义，以及 Theorem12.5.4 的左侧覆盖判据。显示的缺额上界把同一伙伴计数论证中的未覆盖点数量写明。
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_halls_thorem.pdf#page=1|MIT 6.042J Hall's Theorem 讲义]]，PDF1–2／Hall.2–Hall.7：指定侧的单射分配与全部子集条件；讲义用 $E(S)$ 表示本卡的邻居集 $N(S)$。

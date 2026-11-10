---
aliases:
  - 可行匹配与带权最优匹配
  - Hall feasibility and weighted matching
  - Feasibility versus weighted optimality
student_os: knowledge-atom
atom_id: MCS-MATCH-012
atom_type: distinction
status: source-checked
part_of:
  - "[[二分图匹配与 Hall 定理.canvas]]"
requires:
  - "[[Hall 定理]]"
---

<!-- bilingual-en:start -->
# Hall feasibility and weighted optimality
<!-- bilingual-en:end -->

*Hall 条件与带权最优*

<!-- bilingual-en:start -->
Can every student receive a project, and which complete assignment costs least? These are different questions. [[Hall 定理|Hall's theorem]] answers the first from the allowed pairs alone. Edge costs add information needed to compare complete assignments; changing those costs can change the best assignment without changing whether an assignment exists.
<!-- bilingual-en:end -->

每名学生是否都能得到项目，以及哪份完整分配成本最低，是两个不同问题。[[Hall 定理]]只根据允许配对回答前者；边成本另外提供比较完整分配的信息。改变成本可以改变最优分配，却不改变分配是否存在。

<!-- bilingual-en:start -->
## Keep the feasible choices separate from their costs
<!-- bilingual-en:end -->

*把可行选择与成本分开*

<!-- bilingual-en:start -->
Use a finite simple undirected [[二分图|bipartite graph]] $G=(L\cup R,E)$ with specified disjoint sides. An edge says that a pair is allowed. A [[匹配定义|matching]] is a selected subset of edges with no repeated endpoint. Here require it to [[左侧覆盖与完美匹配|cover the entire left side]] $L$; vertices of $R$ may remain unused. Write $\mathcal F_L$ for the set of all matchings meeting that coverage requirement. This is the **feasible set**, meaning all choices that obey the stated rules.
<!-- bilingual-en:end -->

采用有限简单无向[[二分图]] $G=(L\cup R,E)$，已指定互不相交的两侧。一条边表示这对对象允许配对；[[匹配定义|匹配]]则是端点不重复的已选边子集。这里还要求[[左侧覆盖与完美匹配|覆盖左侧]] $L$ 的全部顶点，允许闲置右点。用 $\mathcal F_L$ 表示满足这一覆盖要求的全部匹配，即**可行集合**：所有遵守已声明规则的选择。

<!-- bilingual-en:start -->
For a subset $S\subseteq L$, its neighbor set $N(S)$ contains the right vertices joined to at least one member of $S$. Hall's condition is
<!-- bilingual-en:end -->

对左侧子集 $S\subseteq L$，邻居集 $N(S)$ 包含与 $S$ 中至少一个点相连的全部右点。Hall 条件是

$$
\forall S\subseteq L,\qquad |N(S)|\ge |S|.
$$

<!-- bilingual-en:start -->
The vertical bars count vertices. Hall's theorem says exactly that $\mathcal F_L$ is nonempty if and only if this condition holds. Its input is the edge set: no edge cost appears in $N(S)$. Failure means no left-covering assignment exists under these unit-capacity rules; success does not compare the assignments that do exist.
<!-- bilingual-en:end -->

竖线表示顶点数。Hall 定理恰好说明：$\mathcal F_L$ 非空，当且仅当上述条件成立。它读取的是边集，$N(S)$ 中没有成本。条件失败，表示在这些单位容量规则下没有覆盖左侧的分配；条件成立，却没有比较已有分配的优劣。

<!-- bilingual-en:start -->
Now give each allowed edge $e$ a finite real cost $c(e)$. For a matching $M$, add the costs of its selected edges, with the empty sum equal to zero:
<!-- bilingual-en:end -->

现在给每条允许边 $e$ 一个有限实数成本 $c(e)$。匹配 $M$ 的成本是全部已选边成本之和；空和约定为零：

$$
C(M)=\sum_{e\in M}c(e),\qquad \min_{M\in\mathcal F_L} C(M).
$$

<!-- bilingual-en:start -->
The expression on the right asks for the least cost **among the feasible choices**. When Hall holds, this minimum exists: a finite edge set has only finitely many subsets, so the nonempty set $\mathcal F_L$ supplies a finite nonempty list of real costs, which has a least value. This remains true for zero or negative finite costs. Hall therefore helps establish that this finite optimization problem has a solution, but its condition alone does not identify which matching minimizes the cost.
<!-- bilingual-en:end -->

右边要求在**可行选择之中**找最低成本。Hall 成立时，这个最小值确实存在：有限边集只有有限多个子集，因此非空的 $\mathcal F_L$ 给出有限且非空的一列实数成本，其中有最小值。零成本或负的有限成本不破坏这个论证。所以 Hall 有助于确认这个有限优化问题有解，但仅凭其条件不能确定哪份匹配成本最低。

<!-- bilingual-en:start -->
## The same four edges can have different optima
<!-- bilingual-en:end -->

*相同四条边可以有不同最优解*

<!-- bilingual-en:start -->
Let $L=\{a,b\}$ and $R=\{1,2\}$, and include all four cross-side edges. Each one-vertex subset of $L$ has two neighbors, both left vertices together have two, and the empty subset has none. Hall's condition holds. The only left-covering matchings are the following two perfect matchings; $a-1$ denotes the edge joining $a$ to $1$:
<!-- bilingual-en:end -->

设 $L=\{a,b\}$、$R=\{1,2\}$，四条跨侧边全部存在。左侧单点子集各有两个邻点，两点合起来有两个邻点，空集没有邻点，因此 Hall 条件成立。覆盖左侧的匹配只有下面两种，而且都完美；其中 $a-1$ 表示连接 $a$ 与 $1$ 的边：

$$
\{a-1,b-2\},\qquad \{a-2,b-1\}
$$

> [!example] Swap costs, keep eligibility · 交换成本，保留资格
> <!-- bilingual-en:start -->
> First give $a1,b2$ cost $1$ each, and $a2,b1$ cost $10$ each. The two total costs are $2$ and $20$, so the first matching is uniquely cheapest. Now swap these costs: the totals become $20$ and $2$, and the second matching is uniquely cheapest. The vertices, edges, feasible matchings and every Hall inequality are identical in the two instances. Only the cost data changed.
> <!-- bilingual-en:end -->
>
> 先令 $a1,b2$ 的成本各为 $1$，$a2,b1$ 的成本各为 $10$。两份总成本分别为 $2,20$，第一份是唯一最低成本解。交换这些成本后，总成本变为 $20,2$，第二份成为唯一最低成本解。两实例的点、边、可行匹配及每一个 Hall 不等式都相同，变化的只有成本数据。

<!-- bilingual-en:start -->
This gives a direct reason Hall's condition cannot determine the weighted optimum: any conclusion depending only on $L,R,E$ is unchanged by this swap, while the minimizing edge set changes. If all four costs are $1$, both matchings cost $2$; feasibility also says nothing about uniqueness of the optimum.
<!-- bilingual-en:end -->

这直接说明 Hall 条件为何不能确定带权最优解：任何只依赖 $L,R,E$ 的结论，在换权后都不变，最优边集却改变了。若四条边成本都为 $1$，两份匹配的成本均为 $2$；可行性也没有保证最优解唯一。

<!-- bilingual-en:start -->
## Name the coverage rule and the optimization order
<!-- bilingual-en:end -->

*写明覆盖规则与优化顺序*

<!-- bilingual-en:start -->
Minimizing cost among perfect matchings fixes coverage of every vertex. First maximizing [[极大与最大匹配|matching cardinality]], and then minimizing cost among all matchings of that maximum size, gives priority to the number of assignments. Maximizing total benefit while allowing unmatched vertices need not prioritize assignment count. These are different feasible sets or different orders of preference, even on the same graph.
<!-- bilingual-en:end -->

在完美匹配中最小化成本，先固定了覆盖全部顶点的要求。先最大化[[极大与最大匹配|匹配基数]]，再在最大基数的全部匹配中最小化成本，是先保证分配人数。允许对象未匹配而直接最大化总收益，则未必优先追求人数。同一张图上，这些做法对应不同可行集合，或者不同的比较顺序。

<!-- bilingual-en:start -->
For a concrete distinction, keep only edges $a1,a2,b1$. Give them benefits $100,1,1$, respectively. The matching $\{a2,b1\}$ serves both left vertices and has benefit $2$; $\{a1\}$ serves only one and has benefit $100$. The former maximizes cardinality, whereas the latter maximizes benefit over unrestricted matchings. If full left coverage is mandatory, the one-edge choice is infeasible, regardless of its benefit.
<!-- bilingual-en:end -->

具体地，只保留边 $a1,a2,b1$，收益分别为 $100,1,1$。匹配 $\{a2,b1\}$ 覆盖两个左点，收益为 $2$；$\{a1\}$ 只覆盖一个，收益却为 $100$。前者最大化基数，后者在不要求覆盖的全部匹配中最大化收益。如果覆盖全部左点是硬性要求，一条边的方案无论收益多高都不可行。

<!-- bilingual-en:start -->
Similarly, minimizing positive costs over all matchings without requiring coverage chooses the empty matching, at cost zero. That may solve the written optimization problem while failing the intended assignment task. [[匹配建模要素|State the model requirements]] before interpreting an “optimal” answer.
<!-- bilingual-en:end -->

同样，若所有成本都为正，却在全部匹配中最小化成本而不要求覆盖，空匹配以零成本获胜。它可能正确求解了写下的优化问题，却没有完成原本想做的分配任务。解释“最优”结果之前，先[[匹配建模要素|写清建模要求]]。

> [!warning] Changing weights is different from changing allowed edges · 改权重不同于改允许边
> <!-- bilingual-en:start -->
> The distinction assumes the allowed edge set stays fixed. If a cost limit is used to delete expensive edges, the new neighbor sets can change, so Hall must be checked on the filtered graph. That test asks whether coverage is possible using only edges within the limit; it still does not minimize the sum of their costs. A forbidden pair is absent, not an allowed edge of cost zero.
> <!-- bilingual-en:end -->
>
> 上述区别以允许边集不变为前提。若用成本门槛删去昂贵边，邻居集就可能改变，需要在筛选后的图上重新检查 Hall。这个检查回答能否只用门槛内的边完成覆盖，仍不是最小化成本总和。禁止配对表示没有边，不是存在一条零成本的允许边。

<!-- bilingual-en:start -->
If $L$ is empty, its only left-covering matching in a bipartite graph is empty and costs zero. If Hall fails for nonempty $L$, this particular feasible set is empty: a cheap partial matching cannot be presented as a solution that covers $L$. Weighted optimization and [[稳定匹配与普通匹配|preference stability]] are also separate requirements; summing costs does not test whether two participants would both prefer to switch partners.
<!-- bilingual-en:end -->

若 $L$ 为空，二分图中唯一覆盖左侧的匹配是空匹配，成本为零。若非空 $L$ 不满足 Hall，这个可行集合就是空的；便宜的部分匹配不能冒充覆盖 $L$ 的解。带权优化与[[稳定匹配与普通匹配|偏好稳定性]]也各有要求；成本求和不会检查某两位参与者是否都更愿意改配彼此。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=519|MIT Mathematics for Computer Science, §12.5.2, PDF519–521 / printed511–513]]: Hall's condition, neighbor sets, and coverage of a designated side; the finite-graph convention is on PDF517.
- MIT 6.1200J, [Lecture12: Matching](https://ocw.mit.edu/courses/6-1200j-mathematics-for-computer-science-spring-2024/mit6_1200j_s24_lec12.pdf), PDF2–4, especially Definitions5–6 on3: cardinality, finite real edge weights, their sum, minimum-weight perfect matching and the separate preference data. Finite attainment and all counterexamples above are proved or computed directly.
<!-- bilingual-en:end -->

- MIT《Mathematics for Computer Science》§12.5.2，PDF519–521／印511–513页：Hall 条件、邻居集合与指定一侧覆盖；PDF517 明示有限图约定。
- MIT 6.1200J Lecture12，PDF2–4，尤其第3页 Definitions5–6：匹配基数、实数边权、权重求和、最小权完美匹配及独立的偏好信息。正文另给有限集合取最小值的论证，并直接核验全部反例。

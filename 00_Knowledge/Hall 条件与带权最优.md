---
aliases:
  - "Hall 条件只判断可行性而不决定带权最优匹配"
  - 可行匹配与带权最优匹配
  - Hall condition does not optimise weights
  - Weighted matching boundary
student_os: knowledge-atom
atom_id: MCS-MATCH-012
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[Hall 定理]]"
part_of:
  - "[[二分图匹配与 Hall 定理.canvas]]"
---

# Hall 条件只判断可行性而不决定带权最优匹配

<!-- bilingual-en:start -->
*Hall's condition decides feasibility but not an optimal weighted matching*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> Hall condition 只读取哪些边存在，因此回答“能否覆盖指定一侧”。若每条允许边还有成本、距离或收益，选择总权重最好的一组边是另一个优化问题；Hall 定理不会比较两个都可行的 matchings。
>
> <!-- bilingual-en:start -->
> Hall's condition reads only which edges exist, so it answers whether a designated side can be covered. If eligible edges also carry costs, distances, or benefits, choosing the best total weight is a separate optimisation problem. Hall's theorem does not rank two matchings that are both feasible.
> <!-- bilingual-en:end -->

设 $L=\{a,b\}$、$R=\{1,2\}$，四条边都存在。这张 allowed-edge graph 满足 Hall condition，并有两组可行的 perfect matchings：
$$
\{a-1,b-2\},\qquad \{a-2,b-1\}
$$
若第一组两条边的成本各为 $1$，第二组各为 $10$，最低成本解显然是第一组；交换权重后，allowed-edge graph 与 Hall condition 完全不变，最优解却改变。
<!-- bilingual-en:start -->
Let $L=\{a,b\}$, $R=\{1,2\}$, with all four edges present. This allowed-edge graph satisfies Hall's condition and has the two feasible perfect matchings shown above. If the two edges in the first matching each cost one and the two in the second each cost ten, the first is the minimum-cost solution. Swapping those weights leaves the allowed-edge graph and Hall's condition unchanged while changing the optimum.
<!-- bilingual-en:end -->

建模时还要明确优化顺序。可能的目标包括：在 perfect matchings 中最小化成本；先最大化 cardinality，再在最大 matchings 中最小化成本；或在允许不匹配时直接最大化净收益。它们可能给出不同答案，不能只写“找最优 matching”。
<!-- bilingual-en:start -->
The optimisation order must also be stated. Possible objectives include minimising cost among perfect matchings, first maximising cardinality and then minimising cost among maximum matchings, or directly maximising net benefit while allowing unmatched vertices. These can produce different answers, so “find the optimal matching” is underspecified.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 两张实例的 allowed-edge graph 完全相同，但边权不同。Hall condition 会变化吗？最优 matching 会变化吗？
>
> **答案：** Hall condition 不变；带权最优 matching 可能改变。

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]]，Section 12.5.2：核验 Hall condition 只由 neighbourhood cardinalities 决定。
  <!-- bilingual-en:start -->
  Section 12.5.2 of [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] verifies that Hall's condition depends only on neighbourhood cardinalities.
  <!-- bilingual-en:end -->
- MIT 6.1200J，[Lecture 12: Matching](https://ocw.mit.edu/courses/6-1200j-mathematics-for-computer-science-spring-2024/mit6_1200j_s24_lec12.pdf)，第 3 页：核验 weighted graph、matching weight 与 minimum-weight perfect matching 是在可行边集之上另加的优化问题。
  <!-- bilingual-en:start -->
  MIT 6.1200J, [Lecture 12: Matching](https://ocw.mit.edu/courses/6-1200j-mathematics-for-computer-science-spring-2024/mit6_1200j_s24_lec12.pdf), p. 3, verifies weighted graphs, matching weight, and minimum-weight perfect matching as an optimisation problem added to the feasible edge set.
  <!-- bilingual-en:end -->

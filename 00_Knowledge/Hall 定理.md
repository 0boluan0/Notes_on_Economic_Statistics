---
aliases:
  - "Hall 定理刻画覆盖指定一侧的匹配是否存在"
  - Hall 定理
  - Hall's Marriage Theorem
  - Hall condition
student_os: knowledge-atom
atom_id: MCS-MATCH-008
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[左侧覆盖与完美匹配]]"
part_of:
  - "[[二分图匹配与 Hall 定理.canvas]]"
---

# Hall 定理刻画覆盖指定一侧的匹配是否存在

<!-- bilingual-en:start -->
*Hall's theorem characterises when a matching covers a designated side*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 设 $G=(L\cup R,E)$ 是 finite bipartite graph。对 $S\subseteq L$，定义
> $$
> N(S)=\{r\in R:\text{存在 }\ell\in S\text{ 使 }(\ell,r)\in E\}.
> $$
> 存在覆盖全部 $L$ 的 matching，当且仅当
> $$
> |N(S)|\ge |S|\qquad\text{对每个 }S\subseteq L.
> $$
>
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

必要性来自单射：若 matching 覆盖 $L$，那么 $S$ 中每个左点都匹配到 $N(S)$ 中一个不同的右点，因此 $|S|\le |N(S)|$。充分性更深：只要任何左侧子集都没有挤进更小的共同邻居集，就一定存在某种全局重排覆盖 $L$。
<!-- bilingual-en:start -->
Necessity comes from injectivity: if a matching covers $L$, every vertex of $S$ is assigned to a distinct vertex in $N(S)$, so $|S|\le |N(S)|$. Sufficiency is deeper: if no subset of the left side is crowded into a smaller common neighbourhood, some global rearrangement covers $L$.
<!-- bilingual-en:end -->

定理必须指定被覆盖的一侧。若目标是覆盖 $R$，应交换左右角色并对 $R$ 的所有子集检查条件；若 $|L|=|R|$，覆盖 $L$ 自动成为 perfect matching。权重、容量和偏好不会出现在 Hall 条件里，因此需要另行建模。
<!-- bilingual-en:start -->
The theorem must name the side to be covered. To cover $R$, swap the roles and check every subset of $R$. If $|L|=|R|$, covering $L$ automatically gives a perfect matching. Weights, capacities, and preferences do not appear in Hall's condition and therefore require separate modelling.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 为什么只检查每个单点 $\{\ell\}$ 都有邻居不够？
>
> **答案：** 多个左点可能各自有邻居，却全部共享同一个过小的邻居集合；Hall 条件必须检查所有左侧子集。

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]]，Section 12.5.2：核验 neighbourhood、matching condition 与 Hall 定理的必要充分表述。
  <!-- bilingual-en:start -->
  Section 12.5.2 of [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] verifies neighbourhoods, the matching condition, and the necessary-and-sufficient statement of Hall's theorem.
  <!-- bilingual-en:end -->
- MIT 18.200，[Lecture 11–12: Network Flows and Matching](https://ocw.mit.edu/courses/18-200-principles-of-discrete-applied-mathematics-spring-2024/mit18_200_s24_lec11-12.pdf)，第 8–9 页：用 max-flow/min-cut 交叉核验覆盖指定一侧的 Hall 条件。
  <!-- bilingual-en:start -->
  MIT 18.200, [Lecture 11–12: Network Flows and Matching](https://ocw.mit.edu/courses/18-200-principles-of-discrete-applied-mathematics-spring-2024/mit18_200_s24_lec11-12.pdf), pp. 8–9, cross-checks Hall's condition for covering a designated side through max flow and min cut.
  <!-- bilingual-en:end -->

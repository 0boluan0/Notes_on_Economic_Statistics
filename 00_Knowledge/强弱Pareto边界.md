---
aliases:
  - 强Pareto边界排除无人受损而有人严格获益的替代，弱边界只排除人人严格获益的替代
  - The strong Pareto frontier excludes improvements without harm, while the weak frontier excludes only strict improvements for everyone
  - Strong and weak Pareto frontiers
student_os: knowledge-atom
atom_id: PF-WEL-024
atom_type: distinction
status: source-checked
---

# 强Pareto边界排除无人受损而有人严格获益的替代，弱边界只排除人人严格获益的替代
<!-- bilingual-en:start -->
*The strong Pareto frontier excludes improvements without harm, while the weak frontier excludes only strict improvements for everyone*
<!-- bilingual-en:end -->

在同一个[[效用可能集]]中，强 Pareto 边界排除所有“每个人至少一样好、至少一人严格更好”的可行替代；弱 Pareto 边界只排除“每个人都严格更好”的可行替代。本库的[[效用可能性边界]]采用前者，与既有 [[Pareto改进与有效|Pareto 有效标准]]一致。
<!-- bilingual-en:start -->
Within the same [[效用可能集|utility possibility set]], the strong Pareto frontier rules out feasible alternatives that leave everyone at least as well off and make someone strictly better off. The weak frontier rules out only alternatives that make everyone strictly better off. The vault's [[效用可能性边界|utility possibility frontier]] uses the strong convention, consistent with its existing [[Pareto改进与有效|Pareto-efficiency criterion]].
<!-- bilingual-en:end -->

设候选点为 $v\in\mathcal U$，以不等号核对定义：
<!-- bilingual-en:start -->
For a candidate $v\in\mathcal U$, check the definitions through their inequalities:
<!-- bilingual-en:end -->

$$
\begin{aligned}
P_s(\mathcal U)&=\{v\in\mathcal U:\nexists w\in\mathcal U,
\ w_i\ge v_i\ \forall i,\ w_j>v_j\ \text{for some }j\},\\
P_w(\mathcal U)&=\{v\in\mathcal U:\nexists w\in\mathcal U,
\ w_i>v_i\ \forall i\},\\
P_s(\mathcal U)&\subseteq P_w(\mathcal U).
\end{aligned}
$$

强有效必然弱有效：如果存在让所有人严格获益的替代，它当然也会让至少一人获益且无人受损。反向不成立，因为只改善部分人而保持其余人效用不变，足以推翻强有效，却不能推翻弱有效。
<!-- bilingual-en:start -->
Strong efficiency implies weak efficiency: an alternative that strictly benefits everyone also benefits someone without harming anyone. The converse fails. Improving some people while leaving the others unchanged disproves strong efficiency but not weak efficiency.
<!-- bilingual-en:end -->

一个实际教材例子：两人各有禀赋 $(1,1)$，消费量非负；个人 1 的效用为 $u_1(a,b)=a$，个人 2 为 $u_2(a,b)=b+\sqrt a$。分配 $x^1=(2,1)$、$x^2=(0,1)$ 给出效用 $(2,1)$。个人 1 已持有全部第一种商品，效用不能超过 2，故不可能让两人同时严格变好，原效用向量弱有效。
<!-- bilingual-en:start -->
In a textbook example, each person has endowment $(1,1)$ and consumption is nonnegative. Utilities are $u_1(a,b)=a$ and $u_2(a,b)=b+\sqrt a$. Allocations $x^1=(2,1)$ and $x^2=(0,1)$ yield utilities $(2,1)$. Person 1 holds all of the first good and cannot exceed utility 2, so no alternative can make both people strictly better off. The initial vector is weakly efficient.
<!-- bilingual-en:end -->

但改为 $x^1=(2,0)$、$x^2=(0,2)$，两种商品仍各用 2 单位，效用变为 $(2,2)$。个人 2 严格获益而个人 1 不变，因此原效用向量不是强有效。达到某一个效用坐标的全局上限，只能阻止人人同时严格改善，不能独立证明强有效。
<!-- bilingual-en:start -->
Reallocating to $x^1=(2,0)$ and $x^2=(0,2)$ still uses two units of each good and yields utilities $(2,2)$. Person 2 strictly gains while person 1 is unchanged, so the initial vector is not strongly efficient. Reaching the global maximum in one utility coordinate rules out strict improvement for everyone, but does not by itself establish strong efficiency.
<!-- bilingual-en:end -->

## 来源与核验

- [John Boyd, FIU, *Microeconomics of Competitive Markets*, §21.2.5–21.2.6，PDF 第 15–16 页](https://faculty.fiu.edu/~boydj/microii/microg06-l.pdf#page=15)：强弱效率、不等号与例 21.2.2；集合包含关系由定义直接推得。已核正文计算及实际图页；Figure 21.2.3 上下点的 $\mathbf u^1,\mathbf u^2$ 标签与正文及图注相反，故不复用该图标号。
  <!-- bilingual-en:start -->
  Supports the efficiency distinction, inequalities, and Example 21.2.2. Set inclusion follows directly from the definitions. The calculations and figure page were checked; the two point labels in Figure 21.2.3 are reversed relative to the text and caption and are not reused.
  <!-- bilingual-en:end -->
- [MIT 14.04, Lecture 7 transcript, PDF 第 4 页](https://ocw.mit.edu/courses/14-04-intermediate-microeconomic-theory-fall-2020/1J31KewSsgeY5Yf8L0DVXgVhKBPLTZd_A_transcript.pdf#page=4)：本库前沿采用的“全体弱改善且至少一人严格改善”支配标准。
  <!-- bilingual-en:start -->
  Supports the dominance convention used for the vault's frontier: everyone weakly improves and at least one person strictly improves.
  <!-- bilingual-en:end -->

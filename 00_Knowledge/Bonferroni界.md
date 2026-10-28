---
aliases:
- Bonferroni inequalities
- Bonferroni bounds
- Bonferroni不等式
student_os: knowledge-atom
atom_id: MCS-COUNT-023
atom_set: mcs-counting
atom_type: theorem
status: source-checked
requires:
- '[[容斥原理]]'
part_of:
- '[[组合计数原理.canvas]]'
---

<!-- bilingual-en:start -->
# Bonferroni inequalities
<!-- bilingual-en:end -->
*Bonferroni 界*

<!-- bilingual-en:start -->
Bonferroni inequalities bound the size of a union when we know only some of the overlaps. Adding the individual set sizes gives an upper bound because shared elements are counted repeatedly. Subtracting all pairwise overlaps can overcorrect and gives a lower bound. Further truncations of [[容斥原理|inclusion–exclusion]] alternate between these two directions.

Consider three finite sets $A=\{1,2,3,4\}$, $B=\{3,4,5\}$ and $C=\{4,5,6\}$. Their sizes add to $10$, even though their union has only $6$ elements. The pairwise overlaps have sizes $2,1,2$, whose sum is $5$. Subtracting them gives $10-5=5$, now below the true count. The element $4$ belongs to all three sets: it was counted three times and subtracted three times, so adding the triple overlap restores it. The full count is $10-5+1=6$.
<!-- bilingual-en:end -->
当只知道部分重叠信息时，Bonferroni 界可以约束并集大小。直接相加各集合大小会重复计算共有元素，所以给出上界；减去全部两两交集又可能扣过头，因而给出下界。继续截断[[容斥原理]]的交替和，会在这两个方向之间交替。

例如，三个有限集合为 $A=\{1,2,3,4\}$、$B=\{3,4,5\}$、$C=\{4,5,6\}$。各自大小加起来是 $10$，但并集只有 $6$ 个元素。两两交集的大小分别为 $2,1,2$，合计为 $5$；减去后得到 $10-5=5$，又低于真实数量。元素 $4$ 属于全部三个集合，被加了三次、又减了三次，因此加回三重交集才把它恢复。完整数量是 $10-5+1=6$。

<!-- bilingual-en:start -->
## What an order of overlap means
<!-- bilingual-en:end -->
*一阶、二阶交集分别是什么*

<!-- bilingual-en:start -->
Let $A_1,\ldots,A_m$ be a finite list of finite sets, with integer $m\ge1$. For $j=1,\ldots,m$, let $S_j$ be the sum of the sizes of all intersections formed by choosing $j$ distinct set indices. Thus $S_1$ adds individual sizes, $S_2$ adds each pairwise intersection once, and so on:
<!-- bilingual-en:end -->
设 $A_1,\ldots,A_m$ 是有限个有限集合，整数 $m\ge1$。对 $j=1,\ldots,m$，用 $S_j$ 表示：选出 $j$ 个不同集合索引，对所有这样的交集大小求和。因此，$S_1$ 相加各集合大小，$S_2$ 把每个两两交集加一次，依此类推：

$$
S_j=\sum_{1\le i_1<\cdots<i_j\le m}
|A_{i_1}\cap\cdots\cap A_{i_j}|.
$$

<!-- bilingual-en:start -->
The increasing indices prevent counting one group of sets in multiple orders. Choose a truncation order $r$ with $1\le r\le m$, and keep the first $r$ terms of the alternating sum:
<!-- bilingual-en:end -->
索引递增，是为了不因交换顺序而重复计算同一组集合。选整数截断阶数 $r$，满足 $1\le r\le m$，保留交替和的前 $r$ 项：

$$T_r=S_1-S_2+S_3-\cdots+(-1)^{r-1}S_r.$$

> [!important] Odd above, even below / 奇数阶在上，偶数阶在下
> $$
> \boxed{
> \begin{aligned}
> |A_1\cup\cdots\cup A_m|&\le T_r &&\text{if }r\text{ is odd},\\
> T_r&\le |A_1\cup\cdots\cup A_m| &&\text{if }r\text{ is even}.
> \end{aligned}}
> $$
>
> <!-- bilingual-en:start -->
> In particular, for $m\ge2$, $S_1-S_2\le|\bigcup_i A_i|\le S_1$. At $r=m$, the full inclusion–exclusion sum is exact. For $m=1$, its first term is already exact; an empty list of sets has union size zero.
> <!-- bilingual-en:end -->
> 特别地，$m\ge2$ 时，$S_1-S_2\le|\bigcup_i A_i|\le S_1$。$r=m$ 时，完整容斥和给出精确值。$m=1$ 时，第一项就已精确；若没有任何集合，并集大小为零。

<!-- bilingual-en:start -->
## Why the parity determines the direction
<!-- bilingual-en:end -->
*为什么方向由奇偶性决定*

<!-- bilingual-en:start -->
Follow one element. If it belongs to none of the sets, it contributes zero everywhere. If it belongs to exactly $q\ge1$ of them, it appears in $\binom qj$ of the $j$-fold intersections; this [[组合数|binomial coefficient]] chooses the $j$ containing sets. Its total weight in $T_r$ is therefore
<!-- bilingual-en:end -->
跟踪一个元素。如果它不属于任何集合，各处贡献都是零。如果它恰好属于 $q\ge1$ 个集合，就出现在 $\binom qj$ 个 $j$ 重交集中；这个[[组合数]]是在包含它的集合中选出 $j$ 个。因此，它在 $T_r$ 中的总权重为

$$
w_r(q)=\sum_{j=1}^{\min(r,q)}(-1)^{j-1}\binom qj
=1-(-1)^r\binom{q-1}{r},
$$

<!-- bilingual-en:start -->
where $\binom{q-1}{r}=0$ when $r>q-1$. The last equality follows by applying [[Pascal恒等式|Pascal's identity]] to the alternating sum: the neighbouring terms cancel, leaving only the boundary term. The union should count this element once. For odd $r$, its weight is $1+\binom{q-1}{r}\ge1$; for even $r$, it is $1-\binom{q-1}{r}\le1$. Summing these inequalities over the elements proves the bounds.
<!-- bilingual-en:end -->
这里当 $r>q-1$ 时，约定 $\binom{q-1}{r}=0$。最后一个等式来自对交替和应用 [[Pascal恒等式]]：相邻项抵消，只剩边界项。并集应把这个元素计一次；$r$ 为奇数时，它的权重为 $1+\binom{q-1}{r}\ge1$；$r$ 为偶数时，权重为 $1-\binom{q-1}{r}\le1$。对所有元素求和，就证明了上下界。

<!-- bilingual-en:start -->
This also explains when an early truncation is exact. If no element belongs to more than $r$ sets, every included element already has weight one. For example, three pairwise disjoint sets are counted exactly by $S_1$, even though $r=1<m=3$. Without such information, a truncation is a bound, not an exact count.
<!-- bilingual-en:end -->
这个证明也解释了提前截断何时就能精确：若没有元素同时属于超过 $r$ 个集合，每个被包含的元素权重已经是 1。例如，三个两两不交的集合只用 $S_1$ 就能数准，尽管此时 $r=1<m=3$。没有这样的额外信息时，截断结果是界，不能当作精确计数。

> [!note] Probability form / 概率形式
> <!-- bilingual-en:start -->
> For any finite family of events $A_1,\ldots,A_m$, define $S_j$ using intersection probabilities instead of set sizes. The same odd-upper/even-lower inequalities bound $P(\bigcup_i A_i)$. They hold on an arbitrary probability space and require no independence or equally likely outcomes: the pointwise weight argument above applies to [[事件指示变量|event indicators]], and [[期望线性性|linearity of expectation]] averages it.
>
> If a ticket is drawn uniformly from $1,\ldots,10$ in the opening example, divide each count by $10$: the first two orders give $0.5\le P(A\cup B\cup C)\le1$, and including the triple overlap gives $0.6$. This particular division is valid because the ten tickets are equally likely. A general probability model uses its actual probabilities.
> <!-- bilingual-en:end -->
> 对任意有限个事件 $A_1,\ldots,A_m$，把交集大小改为交集概率来定义 $S_j$，同样的“奇上偶下”不等式就约束 $P(\bigcup_i A_i)$。概率空间可以是任意的，不要求独立或结果等可能：把上面的逐元素权重解释为[[事件指示变量]]，再用[[期望线性性]]取平均即可。
>
> 若在开头例子中，从编号 $1,\ldots,10$ 的签中等概率抽一张，就可以把各项计数除以 $10$。前两阶给出 $0.5\le P(A\cup B\cup C)\le1$，加上三重交集后得到 $0.6$。这里能直接除以 $10$，是因为十张签等可能；一般概率模型要使用实际概率。

<!-- bilingual-en:start -->
Additional intersection information does not guarantee that each new bound is numerically tighter than earlier bounds on the same side. If five sets are all the same singleton, then $S_1=5,S_2=10,S_3=10,S_4=5,S_5=1$, while the union size is one: $T_1=5$ and $T_3=5$ are equally loose. With six copies of that singleton, $T_1=6$ but $T_3=11$, an even looser upper bound. Keep the best applicable upper and lower bounds; probabilities can also be restricted to $[0,1]$.
<!-- bilingual-en:end -->
获得更高阶交集信息，并不保证每个新界都比同侧的旧界更紧。例如，五个集合都等于同一个单元素集合，则 $S_1=5,S_2=10,S_3=10,S_4=5,S_5=1$，并集大小却是 1：$T_1=5$、$T_3=5$ 一样宽松。若有六个这样的集合，$T_1=6$，$T_3=11$，后者反而是更松的上界。应保留已知适用界中最好的上下界；概率还可以结合 $[0,1]$ 的范围。

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 18.310, Homework 6, Problem 2, p. HW6-2 (PDF p. 2)](https://ocw.mit.edu/courses/18-310-principles-of-discrete-applied-mathematics-fall-2013/ba125f616522059788a08e5a38bdb8df_MIT18_310F13_Homework6.pdf): states the intersection-probability sums and both parity directions. This is a problem statement; the pointwise proof is supplied above, not attributed to an unseen solution.
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=697|MIT Mathematics for Computer Science, §15.9.4, printed pp. 689–690 (PDF pp. 697–698)]]: general inclusion–exclusion notation; [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=699|§15.10.1, printed p. 691 (PDF p. 699)]] supplies Pascal's identity used in the weight calculation. All finite examples are enumerated directly.
<!-- bilingual-en:end -->
- MIT 18.310 作业 6，第 2 题，HW6-2 页（PDF 第 2 页）：核验交集概率之和与两个奇偶方向。这是题目陈述；逐点证明由本卡给出，未归因于未读的参考解答。
- MIT《Mathematics for Computer Science》§15.9.4，印刷页 689–690（PDF 页 697–698）：核验一般容斥记号；§15.10.1，印刷页 691（PDF 页 699）：核验权重计算使用的 Pascal 恒等式。所有有限例子另行枚举。

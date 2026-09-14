---
aliases:
  - "容斥交替和截断到奇数阶给并集上界而截断到偶数阶给并集下界"
  - Bonferroni inequalities for truncated inclusion-exclusion
student_os: knowledge-atom
atom_id: MCS-COUNT-023
atom_set: mcs-counting
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[容斥原理]]"
part_of:
  - "[[组合计数原理.canvas]]"
---

# 容斥交替和截断到奇数阶给并集上界而截断到偶数阶给并集下界
<!-- bilingual-en:start -->
*Truncating inclusion-exclusion after an odd order gives an upper bound on the union, while truncating after an even order gives a lower bound*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 对有限集合 $A_1,\ldots,A_m$，令 $S_j$ 为全部 $j$ 重交集大小之和，并令
> $$
> T_r=\sum_{j=1}^{r}(-1)^{j-1}S_j.
> $$
> 若 $r$ 为奇数，则
> $$
> \left|\bigcup_iA_i\right|\le T_r;
> $$
> 若 $r$ 为偶数，则
> $$
> T_r\le\left|\bigcup_iA_i\right|.
> $$
> <!-- bilingual-en:start -->
> Odd truncations of inclusion-exclusion are upper bounds on the union; even truncations are lower bounds.
> <!-- bilingual-en:end -->

最前两级给出熟悉的夹法：
$$
S_1-S_2\le\left|\bigcup_iA_i\right|\le S_1.
$$
每多加入一阶交集，截断值会从另一侧修正此前的过计或欠计。只有写到所有非空交集时，[[容斥原理]] 才给精确等式；提前停止时，可靠结论是带奇偶方向的界，而不是近似等号。

证明可逐元素进行。若某个元素恰属于 $q$ 个集合，它在 $T_r$ 中的权重是
$$
\sum_{j=1}^{\min(r,q)}(-1)^{j-1}\binom qj.
$$
这个部分交替和在奇数 $r$ 时至少为 1，在偶数 $r$ 时至多为 1；对所有元素求和就得到方向。把集合基数替换成事件概率，概率版方向不变。

> [!question]- 自检
> 只知道各集合大小时，$S_1$ 是并集的上界还是下界？再知道两两交集后，$S_1-S_2$ 在哪一侧？
>
> **答案：** $S_1$ 是奇数一阶截断，给上界；$S_1-S_2$ 是偶数二阶截断，给下界。

## 来源与核验

- [MIT OpenCourseWare 18.310, Homework 6](https://ocw.mit.edu/courses/18-310-principles-of-discrete-applied-mathematics-fall-2013/ba125f616522059788a08e5a38bdb8df_MIT18_310F13_Homework6.pdf)：核对 Bonferroni 奇数阶上界与偶数阶下界。
- [[容斥原理]]：复用各阶交集记号及完整交替和。

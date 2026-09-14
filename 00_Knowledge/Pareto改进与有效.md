---
aliases:
  - "Pareto 改进让至少一人严格变好且无人变差"
  - A Pareto improvement makes at least one person strictly better off and no one worse off
  - Pareto 改进与有效
student_os: knowledge-atom
atom_id: MICRO-EX-002
atom_set: exchange-economy-welfare
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[交换经济可行配置]]"
part_of:
  - "[[交换经济、Edgeworth box 与福利定理.canvas]]"
related:
  - "[[互利透镜]]"
  - "[[契约曲线]]"
  - "[[潜在补偿准则]]"
---

# Pareto 改进让至少一人严格变好且无人变差
<!-- bilingual-en:start -->
*A Pareto improvement makes at least one person strictly better off and no one worse off*
<!-- bilingual-en:end -->

> [!summary] 定义
> 从可行配置 $x$ 到另一可行配置 $x'$ 是 Pareto 改进，当且仅当每个人都弱偏好自己的新消费束，并且至少一人严格偏好：
> $$x'_i\succeq_i x_i\ \forall i,\qquad x'_h\succ_h x_h\ \text{for some }h.$$
> 若不存在这样的可行 $x'$，原配置就是 Pareto 有效（Pareto efficient / optimal）。
> <!-- bilingual-en:start -->
> A Pareto improvement is feasible, weakly benefits everyone, and strictly benefits at least one person. An allocation is Pareto efficient when no feasible Pareto improvement exists.
> <!-- bilingual-en:end -->

这个标准比较的是个人自己的偏好排序，不需要把 A 的一单位效用与 B 的一单位效用相加。代价是它对分配判断非常不完备：只要继续帮助一人必然伤害另一人，一个极端不平等配置也可能 Pareto 有效。

“有效”与“改进”必须始终相对于同一个可行集。若技术、总资源、信息或可交易商品集合改变，可行集也会改变，原来有效的配置未必仍有效。Pareto 改进也不等于 [[潜在补偿准则|Kaldor-Hicks 改进]]：后者只要求赢家**有能力**补偿输家，即使补偿没有实际发生。

本卡采用常见约定：“Pareto 有效”表示不存在“人人不差且至少一人严格更好”的可行替代。有些教材另外区分 [[强弱Pareto边界|weak/strong Pareto efficiency]]；遇到这些术语时应先核对其严格与弱不等号，而不是只看中文名称。

> [!question]- 自检
> 把几乎全部资源给 A、只给 B 极少资源，为什么仍可能 Pareto 有效？
>
> **答案：** 若资源已全部使用，而且再增加 B 的福利必然要拿走 A 看重的资源，就不存在无人受损的可行改进。Pareto 有效并不评价这种分配是否公平。

## 来源与核验

- [Stanford ECON 202, *General Equilibrium*](https://web.stanford.edu/~jdlevin/Econ%20202/General%20Equilibrium.pdf)：核对可行配置、Pareto optimality 的弱偏好/严格偏好定义，以及效率不蕴含分配正义。
- [MIT OCW 14.03, Lecture 10](https://ocw.mit.edu/courses/14-03-microeconomic-theory-and-public-policy-fall-2016/e45ec68f98dcb7bcd7529866c0c44dc6_MIT14_03F16_lec10.pdf)：核对“交换收益已经用尽”的图形解释。

---
aliases:
  - "按一个固定元素是否入选分类可得 Pascal 恒等式"
  - Pascal identity by conditioning on one distinguished element
student_os: knowledge-atom
atom_id: MCS-COUNT-018
atom_set: mcs-counting
atom_type: identity
status: source-checked
mastery_state: unassessed
requires:
  - "[[组合数]]"
  - "[[加法法则]]"
related:
  - "[[组合数对称性]]"
  - "[[二项式定理]]"
part_of:
  - "[[组合计数原理.canvas]]"
---

# 按一个固定元素是否入选分类可得 Pascal 恒等式
<!-- bilingual-en:start -->
*Partitioning by whether one distinguished element is selected gives Pascal's identity*
<!-- bilingual-en:end -->

> [!summary] 原子恒等式
> 对 $1\le k\le n-1$，
> $$
> \binom nk=\binom{n-1}{k}+\binom{n-1}{k-1}.
> $$
> 固定一个元素后，每个 $k$ 元子集恰好属于“不选它”或“选它”两类；两类不交且穷尽全部结果。
> <!-- bilingual-en:start -->
> For $1\le k\le n-1$, $\binom nk=\binom{n-1}{k}+\binom{n-1}{k-1}$. Every $k$-subset either omits or contains one fixed element, and these two cases are disjoint and exhaustive.
> <!-- bilingual-en:end -->

不选固定元素时，要从剩余 $n-1$ 个对象中选 $k$ 个；选它时，还要从剩余对象中选 $k-1$ 个。由 [[加法法则]]，两类数量相加得到左边。

若统一约定 $\binom nr=0$ 当 $r<0$ 或 $r>n$，同一公式也覆盖 $k=0$ 与 $k=n$。这项约定只是让边界写法整齐；它不表示真能从 $n$ 个对象中选负数个对象。

> [!question]- 自检
> 用“学生甲是否入选”解释 $\binom{8}{3}=\binom73+\binom72$。
>
> **答案：** 不选甲时从其余 7 人选 3 人；选甲时再从其余 7 人选 2 人。两类互斥且覆盖所有三人组。

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/03_Counting#Pascal 恒等式|Session 26 — Pascal 恒等式]]：核对固定元素分类与完整组合证明。
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf]]：交叉核对 Pascal identity 与边界约定。

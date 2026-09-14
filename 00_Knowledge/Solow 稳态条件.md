---
aliases:
  - "正的唯一 Solow 稳态需要收益递减与端点条件而不由相图交点无条件保证"
  - "A unique positive Solow steady state needs diminishing returns and endpoint conditions"
  - "Solow steady state conditions"
student_os: knowledge-atom
atom_id: MACRO-SOLOW-003
atom_set: solow-growth
atom_type: existence-uniqueness-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Solow 资本积累方程]]"
part_of:
  - "[[Solow 增长模型、稳态与收敛.canvas]]"
implies:
  - "[[Solow 转型动态]]"
  - "[[Solow 黄金律]]"
related:
  - "[[储蓄率的水平效应]]"
---

# 正的唯一 Solow 稳态需要收益递减与端点条件而不由相图交点无条件保证
<!-- bilingual-en:start -->
*A unique positive Solow steady state needs diminishing returns and endpoint conditions; a drawn intersection is not an unconditional guarantee*
<!-- bilingual-en:end -->

> [!summary] 原子条件
> 正的稳态 $k^*>0$ 满足
> $$s f(k^*)=(\delta+n+g)k^*.$$
> 连续、递增、严格凹的 $f$ 配合适当端点条件（标准充分条件是 Inada 条件），使 $f(k)/k$ 从足够高降到足够低，因而与 $(\delta+n+g)/s$ 恰交一次。若这些条件失败，正稳态可能不存在或不唯一。
> <!-- bilingual-en:start -->
> An interior steady state solves $sf(k^*)=(\delta+n+g)k^*$. Continuity, strict concavity, and suitable endpoint behavior make $f(k)/k$ cross the break-even ratio once. Without those restrictions, an interior steady state may fail to exist or may be nonunique.
> <!-- bilingual-en:end -->

标准证明分成两步：

1. **存在：** 若 $\lim_{k\downarrow0}f(k)/k=\infty$、$\lim_{k\to\infty}f(k)/k=0$，连续性与介值定理保证有正解；
2. **唯一：** 严格凹性使平均产出 $f(k)/k$ 严格下降，因此同一 break-even 比率最多被命中一次。

若 $f(0)=0$，$k=0$ 还可能是边界稳态；这里说的“唯一”是唯一的**正内点**稳态。门槛外部性、非凹技术或缺少端点条件都可能产生多个稳态或无正稳态，所以不能把常见相图的形状当成无条件定理。

对 $f(k)=k^\alpha$、$0<\alpha<1$，
$$
k^*=\left(\frac{s}{n+g+\delta}\right)^{\frac{1}{1-\alpha}}.
$$
旧页例子 $\alpha=1/3$、$s=0.24$、$n+g+\delta=0.06$ 给出 $k^*=8$、$y^*=2$；该数值成立是因为 Cobb–Douglas 已满足上述关键形状条件。

> [!question]- 自检
> “$sf(k)$ 是凹的，所以一定有唯一正稳态”少了什么？
>
> **答案：** 还要保证曲线在零附近相对 break-even 线足够高、在远处足够低，并排除非严格凹或门槛导致的多次交点；凹的图形标签本身不证明存在与唯一。

## 来源与核验

- [[02_Economy/10_发展经济学/发展经济学拍屏ppt.pdf#page=72|发展经济学课程 PDF p. 72]]：核对课程采用的标准相图与稳态等式。
- MIT 14.452，[The Solow Growth Model, Lectures 2–3](https://ocw.mit.edu/courses/14-452-economic-growth-fall-2016/2b68057aa4e74410d00ae89a0c49752f_MIT14_452F16_Lec2and3.pdf)，pp. 12–13、28–31、50–53：核对 Inada 条件、内点稳态的存在唯一性证明及条件失效时的反例边界。
- 已从稳态等式重算 Cobb–Douglas 例子 $k^*=8$、$y^*=2$。

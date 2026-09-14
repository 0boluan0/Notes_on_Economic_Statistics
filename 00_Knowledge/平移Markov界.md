---
aliases:
  - "已知下界时平移随机变量可加强 Markov 界"
  - Shifted Markov inequality
  - Shifted Markov bound
  - 平移 Markov 界
student_os: knowledge-atom
atom_id: PROB-CONC-004
atom_set: probability-concentration
atom_type: procedure
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov不等式]]"
part_of:
  - "[[概率不等式与集中界.canvas]]"
---

# 已知下界时平移随机变量可加强 Markov 界
<!-- bilingual-en:start -->
*A known lower bound can strengthen Markov through a shift*
<!-- bilingual-en:end -->

> [!summary] 原子规则
> 若 $X\ge b$ almost surely、$E[X]<\infty$ 且目标阈值 $a>b$，则对非负变量 $X-b$ 应用 Markov：
> $$P(X\ge a)\le\frac{E[X]-b}{a-b}.$$
> <!-- bilingual-en:start -->
> If $X\ge b$ almost surely, $E[X]<\infty$, and the target threshold satisfies $a>b$, apply Markov to the non-negative variable $X-b$:
> $$P(X\ge a)\le\frac{E[X]-b}{a-b}.$$
> <!-- bilingual-en:end -->

事件没有改变：$X\ge a$ 当且仅当 $X-b\ge a-b$。改变的是 Markov 所看到的“非负质量预算”：numerator 从 $E[X]$ 变为 $E[X]-b$，threshold 从 $a$ 变为 $a-b$。当 $b>0$ 且这个下界提供了有效信息时，前者确实减少；若 $b\le0$，不能笼统声称 numerator 变小，价值可能只是让原本可取负值的 $X-b$ 满足非负前提。
<!-- bilingual-en:start -->
The event is unchanged: $X\ge a$ if and only if $X-b\ge a-b$. What changes is the non-negative mass budget seen by Markov: the numerator becomes $E[X]-b$ and the threshold becomes $a-b$. A positive informative lower bound reduces the numerator. When $b\le0$, the numerator need not shrink; the shift may instead be valuable because it makes $X-b$ non-negative when direct Markov on $X$ is unavailable.
<!-- bilingual-en:end -->

例如 $E[X]=150$、$X\ge100$，要界 $P(X\ge200)$。直接 Markov 给 $150/200=3/4$；平移后给

$$
P(X\ge200)\le\frac{150-100}{200-100}=\frac12.
$$

这个改进只来自已确认的支持集信息。
<!-- bilingual-en:start -->
For example, suppose $E[X]=150$ and $X\ge100$, and the target is $P(X\ge200)$. Direct Markov gives $150/200=3/4$, whereas shifting gives
$$P(X\ge200)\le\frac{150-100}{200-100}=\frac12.$$
The improvement comes entirely from the verified support information.
<!-- bilingual-en:end -->

$b$ 必须是真正的 almost-sure lower bound，不能拿样本最小值、分位数或“通常不会更低”代替。随意选择一个过高的 $b$ 会让 $X-b$ 仍可能为负，从而失去 Markov 的前提。类似地，若知道 $X\le u$ 并想控制左尾，可对 $u-X\ge0$ 使用 Markov。
<!-- bilingual-en:start -->
The value $b$ must be a genuine almost-sure lower bound, not a sample minimum, quantile, or value that is merely “usually” not crossed. Choosing $b$ too high leaves $X-b$ potentially negative and destroys Markov's premise. Analogously, if $X\le u$ and a lower tail is of interest, Markov can be applied to $u-X\ge0$.
<!-- bilingual-en:end -->

> [!question]- 自检
> 已知 $P(X\ge0)=0.99$，能否把 $b=0$ 当作下界使用 shifted Markov？
> <!-- bilingual-en:start -->
> If $P(X\ge0)=0.99$, may $b=0$ be used as the lower bound in shifted Markov?
> <!-- bilingual-en:end -->
>
> **答案：** 不能。所需条件是 $P(X\ge b)=1$；剩余的 $1\%$ 负值已经足以破坏非负性前提。
> <!-- bilingual-en:start -->
> **Answer:** No. The required condition is $P(X\ge b)=1$; the remaining one per cent of negative values is enough to violate non-negativity.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] 第 20.1.2 节：推导已知下界 $b$ 时对 $R-b$ 应用 Markov 所得到的改进。
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session33.pdf|MIT 6.042J Session 33]] 第 19.1.2 节与 [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MarkovBounds.pdf|Markov Bounds slides]]：核对课程例子以及下界必须几乎必然成立这一条件。
<!-- bilingual-en:start -->
- Section 20.1.2 of [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] derives the improvement obtained by applying Markov to $R-b$ for a known lower bound $b$.
- Section 19.1.2 of [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session33.pdf|MIT 6.042J Session 33]] and the [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MarkovBounds.pdf|Markov Bounds slides]] verify the course examples and the almost-sure lower-bound requirement.
<!-- bilingual-en:end -->

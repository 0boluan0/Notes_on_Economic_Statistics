---
aliases:
  - "Markov 不等式用非负随机变量的均值控制右尾"
  - Markov's inequality
  - Markov inequality
  - 马尔可夫不等式
student_os: knowledge-atom
atom_id: PROB-CONC-003
atom_set: probability-concentration
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[期望]]"
part_of:
  - "[[概率不等式与集中界.canvas]]"
related:
  - "[[平移Markov界]]"
  - "[[Chebyshev不等式]]"
  - "[[指数Markov法]]"
---

# Markov 不等式用非负随机变量的均值控制右尾
<!-- bilingual-en:start -->
*Markov's inequality controls the upper tail of a non-negative variable by its mean*
<!-- bilingual-en:end -->

> [!summary] 定理
> 若 $X\ge0$ almost surely、$E[X]<\infty$ 且 $a>0$，则
> $$P(X\ge a)\le \frac{E[X]}{a}.$$
> 它只使用非负性和均值，不要求独立、方差或具体分布。
> <!-- bilingual-en:start -->
> If $X\ge0$ almost surely, $E[X]<\infty$, and $a>0$, then
> $$P(X\ge a)\le \frac{E[X]}{a}.$$
> The inequality uses only non-negativity and the mean; it requires neither independence, a variance, nor a specified distribution.
> <!-- bilingual-en:end -->

令 $I=I_{\{X\ge a\}}$。逐个 outcome 都有

$$
X\ge aI,
$$

因为在事件上 $X\ge a$，事件外右侧为 0 而 $X\ge0$。取期望得到 $E[X]\ge aP(X\ge a)$，再除以正数 $a$。
<!-- bilingual-en:start -->
Let $I=I_{\{X\ge a\}}$. Pointwise, $X\ge aI$: on the event, $X\ge a$, while off the event the right-hand side is zero and $X\ge0$. Taking expectations yields $E[X]\ge aP(X\ge a)$; division by the positive number $a$ gives the result.
<!-- bilingual-en:end -->

非负性正是这个逐点比较成立的原因。若 $X$ 可取负值，负尾可以压低 $E[X]$，却不消除正尾；因此不能把“均值很小”直接解释为“右尾很小”。例如 $X=100$ 的概率为 $0.1$、$X=-100/9$ 的概率为 $0.9$ 时 $E[X]=0$，但 $P(X\ge100)=0.1$。
<!-- bilingual-en:start -->
Non-negativity is exactly what makes the pointwise comparison work. If $X$ may be negative, a negative tail can reduce $E[X]$ without removing the positive tail; a small mean alone then does not imply a small upper tail. For example, if $X=100$ with probability $0.1$ and $X=-100/9$ with probability $0.9$, then $E[X]=0$ but $P(X\ge100)=0.1$.
<!-- bilingual-en:end -->

对非负整数计数 $N$，取 $a=1$ 得 $P(N\ge1)\le E[N]$。它说明 expected count 很小时“至少出现一次”也少见，但通常只是上界，不是等式。
<!-- bilingual-en:start -->
For a non-negative integer count $N$, taking $a=1$ gives $P(N\ge1)\le E[N]$. Thus a small expected count makes at least one occurrence unlikely, but the relation is generally an upper bound rather than an equality.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么 $a$ 必须为正，且 $X\ge0$ 不能只替换成 $E[X]\ge0$？
> <!-- bilingual-en:start -->
> Why must $a$ be positive, and why can $X\ge0$ not be replaced merely by $E[X]\ge0$?
> <!-- bilingual-en:end -->
>
> **答案：** 需要除以 $a>0$ 且保持不等号方向；更重要的是证明用到逐点关系 $X\ge aI_{\{X\ge a\}}$，仅有非负均值不能阻止负值在事件外破坏这个关系。
> <!-- bilingual-en:start -->
> **Answer:** Division by $a>0$ must preserve the inequality. More importantly, the proof uses the pointwise relation $X\ge aI_{\{X\ge a\}}$; a non-negative mean alone does not prevent negative values from violating it off the event.
> <!-- bilingual-en:end -->

**继续：** 若还知道 $X$ 的确定下界，[[平移Markov界]]；把 Markov 用在平方偏差上则得到 [[Chebyshev不等式]]。
<!-- bilingual-en:start -->
**Continue with:** A deterministic lower bound leads to [[平移Markov界|a shifted Markov bound]]. Applying Markov to a squared deviation gives [[Chebyshev不等式|Chebyshev's inequality]].
<!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] 定理 20.1.1 与推论 20.1.2：核对非负条件、正阈值条件、证明以及相对阈值形式。
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MarkovBounds.pdf|MIT 6.042J Markov Bounds slides]]：核对课程采用的表述以及非负条件的作用。
<!-- bilingual-en:start -->
- Theorem 20.1.1 and Corollary 20.1.2 of [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] verify the non-negativity condition, positive-threshold condition, proof, and relative-threshold form.
- The [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MarkovBounds.pdf|MIT 6.042J Markov Bounds slides]] verify the course formulation and the role of non-negativity.
<!-- bilingual-en:end -->

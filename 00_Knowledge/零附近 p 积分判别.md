---
aliases:
  - "端点 p 积分从零到一当且仅当 p 小于一收敛"
  - p-integral near zero
  - 零附近 p 积分判别
student_os: knowledge-atom
atom_id: CALC-IMP-005
atom_set: improper-integrals
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[反常积分]]"
part_of:
  - "[[反常积分.canvas]]"
contrasts_with:
  - "[[无穷远 p 积分判别]]"
---

# 端点 p 积分从零到一当且仅当 p 小于一收敛
<!-- bilingual-en:start -->
*The endpoint $p$-integral from zero to one converges exactly when $p<1$*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> $$\int_0^1 x^{-p}\,dx\text{ 收敛}\iff p<1.$$
> 当 $p<1$ 时值为 $1/(1-p)$；$p=1$ 对数发散；$p>1$ 时奇点过强。这个条件与无穷远的 $p>1$ 恰好相反。
> <!-- bilingual-en:start -->
>
> &nbsp;
> $\int_0^1 x^{-p}\,dx$ converges exactly when $p<1$. For $p<1$ its value is $1/(1-p)$; $p=1$ diverges logarithmically, and $p>1$ has too strong a singularity. This criterion is the reverse of the $p>1$ condition at infinity.
> <!-- bilingual-en:end -->

从定义出发，
$$\int_0^1x^{-p}dx=\lim_{\varepsilon\to0^+}\int_\varepsilon^1x^{-p}dx.$$
若 $p\ne1$，右侧等于
$$\lim_{\varepsilon\to0^+}\frac{1-\varepsilon^{1-p}}{1-p}.$$
只有 $1-p>0$ 时，$\varepsilon^{1-p}\to0$，得到有限值。
<!-- bilingual-en:start -->
By definition, $\int_0^1x^{-p}dx=\lim_{\varepsilon\to0^+}\int_\varepsilon^1x^{-p}dx$. For $p\ne1$, this becomes $\lim_{\varepsilon\to0^+}(1-\varepsilon^{1-p})/(1-p)$, which is finite only when $1-p>0$ and $\varepsilon^{1-p}\to0$.
<!-- bilingual-en:end -->

## 为什么两端条件相反

增大 $p$ 会让 $x^{-p}$ 在 $x\to0^+$ 时爆得更快，却让它在 $x\to\infty$ 时降得更快。同一代数形式必须先问“正在逼近哪个问题点”，再选临界条件；不能只背“$p$ 大于一收敛”。
<!-- bilingual-en:start -->
Increasing $p$ makes $x^{-p}$ blow up faster as $x\to0^+$ but decay faster as $x\to\infty$. For the same algebraic form, first identify the problematic limit before choosing the criterion. Memorising only “$p>1$ converges” confuses the two ends.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么 $\int_0^1x^{-3/4}dx$ 收敛，而 $\int_1^\infty x^{-3/4}dx$ 发散？
>
> **答案：** 在零附近用 $p<1$，所以收敛；在无穷远用 $p>1$，而 $3/4$ 不满足，所以发散。

## 来源与核验

- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses93a_Lecture_Notes.pdf|MIT 18.01SC Session 93a]]：核对 $1/\sqrt{x}$、$1/x$、$1/x^2$ 在零附近的例子。
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses93c_Lecture_Notes.pdf|MIT 18.01SC Session 93c]]：核对零附近与无穷远的相反比较方向。
- [[01_Math/01_calculus/05_Infinite_Series_and_Improper_Integrals.md#有限端点奇点|课程有限端点模型]]：核对一般 $p$ 的完整分段结论与有限截断定义。
<!-- bilingual-en:start -->
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses93a_Lecture_Notes.pdf|MIT 18.01SC Session 93a]] was checked for the examples $1/\sqrt{x}$, $1/x$, and $1/x^2$ near zero.
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses93c_Lecture_Notes.pdf|MIT 18.01SC Session 93c]] was checked for the reversed comparison direction near zero and at infinity.
- [[01_Math/01_calculus/05_Infinite_Series_and_Improper_Integrals.md#有限端点奇点|The course endpoint model]] was checked for the complete piecewise conclusion for general $p$ and the finite-truncation definition.
<!-- bilingual-en:end -->

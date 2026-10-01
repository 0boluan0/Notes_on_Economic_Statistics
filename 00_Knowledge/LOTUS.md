---
student_os: knowledge-atom
atom_id: c6276dc5-7afe-4a2a-bea8-866ed6b6adb8
status: source-checked
aliases:
  - "可积函数的期望可以直接按原随机变量的分布加权"
  - "Law of the unconscious statistician"
---

# 可积函数的期望可以直接按原随机变量的分布加权

<!-- bilingual-en:start -->
*A transformed expectation can be computed from the original distribution*
<!-- bilingual-en:end -->

若 $g$ 可测且 $E|g(X)|<\infty$，则
$$E[g(X)]=\int g(x)\,dF_X(x).$$
离散时为 $\sum_xg(x)p_X(x)$；有密度时为 $\int g(x)f_X(x)dx$。无需先求 $g(X)$ 的分布；权重仍由 $X$ 的分布提供。

<!-- bilingual-en:start -->
For measurable g with an integrable transform, average g(x) under the original distribution. Use a PMF sum or density integral when available; the transformed distribution is unnecessary.
<!-- bilingual-en:end -->
例如 $f_X(x)=3x^2$（$0<x<1$），则 $E[X^2]=\int_0^1x^2(3x^2)dx=3/5$。这里被平均的值是 $x^2$，权重是 $3x^2dx$。该规则不要求 $g$ 单调；也不允许一般地把结果写成 $g(E[X])$。

<!-- bilingual-en:start -->
For density 3x² on the unit interval, weighting x² gives 3/5. LOTUS does not require monotonicity and does not justify replacing the result by the function evaluated at the mean.
<!-- bilingual-en:end -->

**关联：** [[期望]] · [[随机变量的分布变换]]

## 来源与核验

- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/04_PSI/Lectures/Lecture 2 - Statistics I.pdf#page=12|EC400 PSI Lecture 2，slide 12]]：支持连续 LOTUS；slide 13 支持离散形式，slide 42 给可积条件。

<!-- bilingual-en:start -->
- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/04_PSI/Lectures/Lecture 2 - Statistics I.pdf#page=12|EC400 PSI Lecture 2, slide 12]]: Supports LOTUS; slides 13 and 42 give the discrete form and integrability condition.
<!-- bilingual-en:end -->

- [[04_PSI Lecture 2 - 随机变量、条件分布与独立性|EC400 PSI Lecture 2 正式笔记]]：保留本课的完整算例与讲解语境。

<!-- bilingual-en:start -->
- [[04_PSI Lecture 2 - 随机变量、条件分布与独立性|EC400 PSI Lecture 2 course note]] retains the full worked examples and lecture context.
<!-- bilingual-en:end -->

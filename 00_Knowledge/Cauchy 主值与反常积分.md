---
aliases:
  - "Cauchy 主值协调两侧截断而普通反常积分要求各侧独立收敛"
  - Cauchy principal value versus improper integral
  - Cauchy 主值与反常积分
student_os: knowledge-atom
atom_id: CALC-IMP-010
atom_set: improper-integrals
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[反常积分逐坏点拆分]]"
part_of:
  - "[[反常积分.canvas]]"
---

# Cauchy 主值协调两侧截断而普通反常积分要求各侧独立收敛
<!-- bilingual-en:start -->
*A Cauchy principal value coordinates two truncations, whereas an ordinary improper integral requires independent one-sided convergence*
<!-- bilingual-en:end -->

> [!summary] 原子区别
> 普通反常积分在内部奇点两侧分别取极限，并要求两侧都是有限值。Cauchy principal value（柯西主值）则规定两侧按同一尺度协调截断，再取组合后的一个极限。主值存在不代表普通反常积分收敛。
> <!-- bilingual-en:start -->
>
> &nbsp;
> An ordinary improper integral takes separate one-sided limits at an interior singularity and requires each to be finite. A Cauchy principal value coordinates the approach on the two sides and takes one limit of their combination. Existence of the principal value does not imply convergence of the ordinary improper integral.
> <!-- bilingual-en:end -->

以 $1/x$ 为例，两个单侧截断极限分别满足
$$\lim_{\varepsilon\to0^+}\int_{-1}^{-\varepsilon}\frac{dx}{x}=-\infty,\qquad
\lim_{\varepsilon\to0^+}\int_{\varepsilon}^{1}\frac{dx}{x}=+\infty.$$
所以两个单侧反常积分都发散，普通反常积分也发散。可是对称截断给出
$$\operatorname{PV}\int_{-1}^{1}\frac{dx}{x}
:=\lim_{\varepsilon\to0^+}
\left(\int_{-1}^{-\varepsilon}\frac{dx}{x}
+\int_{\varepsilon}^{1}\frac{dx}{x}\right)=0.$$
零来自强制对称抵消，不是两个单侧积分各自存在。
<!-- bilingual-en:start -->
For $1/x$, the left and right truncated integrals tend to $-\infty$ and $+\infty$, respectively. Thus both one-sided improper integrals diverge, and so does the ordinary improper integral. Yet the symmetric principal value is zero. That zero comes from enforced symmetric cancellation, not from two separately existing one-sided integrals.
<!-- bilingual-en:end -->

## 两种对象服务于不同问题

主值在奇异积分、Hilbert transform 和某些物理对称模型中有独立用途；普通反常积分则描述不依赖截断协调方式的累计值。题目若只写“反常积分”，默认不能偷偷改成主值。需要主值时应显式写 `PV` 并说明采用的截断规则。
<!-- bilingual-en:start -->
Principal values have legitimate uses in singular integrals, the Hilbert transform, and symmetric physical models. Ordinary improper integrals describe accumulation that does not depend on coordinating truncations. If a problem asks for an improper integral, it cannot silently be replaced by a principal value. Write `PV` and state the truncation rule when that is the intended object.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么“左右都是无穷但正负抵消成零”在普通反常积分中不是合法算术？
>
> **答案：** $+\infty$ 和 $-\infty$ 不是可相加的实数；普通定义先要求两侧各自有有限极限，条件未满足就没有总和。

## 来源与核验

- [[01_Math/01_calculus/05_Infinite_Series_and_Improper_Integrals.md#区间内部奇点|课程内部奇点说明]]：核对两侧独立收敛与 Cauchy 主值的区别。
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses93a_Lecture_Notes.pdf|MIT 18.01SC Session 93a]]：核对直接跨越奇点套原函数会得到荒谬结果。
<!-- bilingual-en:start -->
- [[01_Math/01_calculus/05_Infinite_Series_and_Improper_Integrals.md#区间内部奇点|The course treatment of interior singularities]] was checked for separate one-sided convergence versus Cauchy principal value.
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses93a_Lecture_Notes.pdf|MIT 18.01SC Session 93a]] was checked for the absurd result obtained by applying one antiderivative formula directly across a singularity.
<!-- bilingual-en:end -->

---
aliases:
  - "在二阶连续可微的驻点处，不定 Hessian 推出鞍点，因为任意小邻域内同时存在上升与下降方向"
  - An indefinite Hessian at a stationary point implies a saddle point
student_os: knowledge-atom
atom_id: CALC-MV-011
atom_set: multivariable-differentiation
atom_type: diagnostic-rule
status: source-checked
mastery_state: unassessed
requires:
  - "[[Hessian矩阵]]"
  - "[[多元Taylor近似]]"
related:
  - "[[Hessian 局部极小判据]]"
  - "[[Hessian 局部极大判据]]"
  - "[[半定Hessian无结论]]"
part_of:
  - "[[多元微分.canvas|多元微分]]"
---

# 在二阶连续可微的驻点处，不定 Hessian 推出鞍点，因为任意小邻域内同时存在上升与下降方向
<!-- bilingual-en:start -->
*At a $C^2$ stationary point, an indefinite Hessian implies a saddle because every sufficiently small neighbourhood contains both ascent and descent directions*
<!-- bilingual-en:end -->

> [!summary] 核心判据
> 设 $f$ 在 $x_*$ 的邻域为 $C^2$，且 $\nabla f(x_*)=0$。若 $\nabla^2f(x_*)$ 不定，即存在非零向量 $u,v$ 使
> $$
> u^T\nabla^2f(x_*)u>0,
> \qquad
> v^T\nabla^2f(x_*)v<0,
> $$
> 则 $x_*$ 是鞍点：任意充分小的邻域内，都有函数值高于和低于 $f(x_*)$ 的点。
> <!-- bilingual-en:start -->
> Let $f$ be $C^2$ near a stationary point $x_*$. If its Hessian is indefinite there, so that the displayed quadratic form is positive along one nonzero direction and negative along another, then every sufficiently small neighbourhood contains values both above and below $f(x_*)$. Hence $x_*$ is a saddle point.
> <!-- bilingual-en:end -->

## Taylor 展开为什么锁定两个相反符号
<!-- bilingual-en:start -->
*Why Taylor expansion fixes two opposite signs*
<!-- bilingual-en:end -->

沿 $u$ 取很小的 $t\ne0$。因为一阶项在驻点消失，[[多元Taylor近似]] 给出
$$
f(x_*+tu)-f(x_*)
=\frac{t^2}{2}u^T\nabla^2f(x_*)u+o(t^2).
$$
二次项的系数严格为正，所以当 $t$ 足够小时，余项不能改变其符号。沿 $v$ 的同一展开具有严格为负的二次项，因此在同样小的尺度上得到下降点。两条方向共同排除了局部最大和局部最小。
<!-- bilingual-en:start -->
Along $u$, the first-order term vanishes because $x_*$ is stationary, leaving the displayed second-order Taylor expansion. Its quadratic coefficient is strictly positive, so the remainder cannot reverse the sign for sufficiently small nonzero $t$. The same expansion along $v$ has a strictly negative quadratic coefficient and produces nearby descent points. The two directions together rule out both a local maximum and a local minimum.
<!-- bilingual-en:end -->

## 最小例子
<!-- bilingual-en:start -->
*Minimal example*
<!-- bilingual-en:end -->

对
$$
f(x,y)=x^2-y^2,
$$
原点是驻点，Hessian 为 $\operatorname{diag}(2,-2)$。沿 $x$ 轴函数值高于 $f(0,0)$，沿 $y$ 轴函数值低于 $f(0,0)$，所以原点是鞍点。
<!-- bilingual-en:start -->
For $f(x,y)=x^2-y^2$, the origin is stationary and the Hessian is $\operatorname{diag}(2,-2)$. Values are above $f(0,0)$ along the $x$ axis and below it along the $y$ axis, so the origin is a saddle.
<!-- bilingual-en:end -->

## 判据边界
<!-- bilingual-en:start -->
*Boundary of the test*
<!-- bilingual-en:end -->

不定性是推出鞍点的充分条件，但不是鞍点的必要条件。若 Hessian 只有半定性，二阶项会在某些方向消失，高阶项仍可能产生极值或鞍点；这时本判据没有结论，具体对照见 [[半定Hessian无结论]]。同样，“Hessian 不是正定”也不能直接推出鞍点，因为负定 Hessian 对应局部极大。
<!-- bilingual-en:start -->
Indefiniteness is sufficient for a saddle but is not necessary. If the Hessian is merely semidefinite, its quadratic term vanishes in some directions and higher-order terms may still create an extremum or a saddle; this test is then inconclusive, as shown in [[半定Hessian无结论|the semidefinite-Hessian boundary]]. Likewise, a Hessian that is not positive definite need not indicate a saddle, because a negative-definite Hessian corresponds to a local maximum.
<!-- bilingual-en:end -->

> [!question]- 自检
> 已知 $x_*$ 是 $C^2$ 函数的驻点，且 $\nabla^2f(x_*)$ 有一个正特征值和一个负特征值。可以作出什么局部结论？为什么 $o(\|h\|^2)$ 余项不会推翻它？
> <!-- bilingual-en:start -->
> Suppose $x_*$ is a stationary point of a $C^2$ function and its Hessian has one positive and one negative eigenvalue. What local conclusion follows, and why can the $o(\|h\|^2)$ remainder not overturn it?
> <!-- bilingual-en:end -->
>
> **答案：** Hessian 不定，所以 $x_*$ 是鞍点。沿对应的两个特征向量，严格正负的二次项都是 $t^2$ 量级，而余项相对 $t^2$ 趋于零，足够小时不能改变两者的符号。
> <!-- bilingual-en:start -->
> **Answer:** The Hessian is indefinite, so $x_*$ is a saddle. Along the corresponding eigenvectors, the strictly positive and negative quadratic terms are of order $t^2$, while the remainder is negligible relative to $t^2$ and therefore cannot reverse either sign for sufficiently small $t$.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- LSE EC400，*SOFP Slides Lecture 2*（课程讲义） slides 10–14：核对驻点处不定 Hessian 的鞍点分类，以及半定情形下二阶检验无结论的边界。
  <!-- bilingual-en:start -->
  *English:* EC400 SOFP Slides Lecture 2, slides 10–14, supports the saddle classification for an indefinite Hessian at a stationary point and the inconclusive semidefinite boundary.
  <!-- bilingual-en:end -->
- [[多元Taylor近似]]：核对驻点处二次项和 $o(\|h\|^2)$ 余项如何保持严格正负方向。
  <!-- bilingual-en:start -->
  *English:* [[多元Taylor近似|The multivariable Taylor approximation]] supports the sign argument based on the quadratic term and the $o(\|h\|^2)$ remainder at a stationary point.
  <!-- bilingual-en:end -->
- [[半定Hessian无结论]]：核对不定充分条件不是必要条件，以及半定 Hessian 需要继续检查高阶项的边界。
  <!-- bilingual-en:start -->
  *English:* [[半定Hessian无结论|The semidefinite-Hessian boundary]] verifies that indefiniteness is not necessary for a saddle and that higher-order terms must be examined in semidefinite cases.
  <!-- bilingual-en:end -->

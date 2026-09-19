---
student_os: knowledge-atom
atom_id: OPT-TOOLS-016
aliases:
  - "Hessian 处处负定足以保证严格凹而严格凹不要求 Hessian 处处负定"
  - "An everywhere negative-definite Hessian implies strict concavity but is not necessary for it"
status: needs-review
---

# Hessian 处处负定足以保证严格凹而严格凹不要求 Hessian 处处负定

<!-- bilingual-en:start -->
*An everywhere negative-definite Hessian implies strict concavity but is not necessary for it*
<!-- bilingual-en:end -->

在开凸域上，$f\in C^2$ 且 $H_f(x)\prec0$ 处处成立，则每条非平凡线段限制的二阶导数严格负，因此 $f$ 严格凹。

反向失败：$f(x)=-x^4$ 严格凹，因其导数 $-4x^3$ 严格递减，但 $f''(0)=0$。严格凹是任意不同端点的严格弦不等式，不是二阶导数处处严格负的另一种名称。普通凹性用[[Hessian凹凸性判据]]；最优解的取得性由[[严格凹不保证解存在]]区分。

<!-- bilingual-en:start -->
Strictly negative directional second derivatives on every nontrivial segment imply strict concavity. The function $-x^4$ disproves the converse: its derivative is strictly decreasing but its second derivative vanishes at zero. Ordinary concavity uses the [[Hessian凹凸性判据|semidefinite criterion]], and [[严格凹不保证解存在|existence remains a separate issue]].
<!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/08_MathsCamp-EC400/2026_Course_Materials/02_SOFP/Lectures/EC400 Slides Lecture 1.pdf#page=49|EC400 SOFP 原材料，PDF p.49]]：课件提供普通凹性的 Hessian 判据；负定的充分性沿线段推导，严格凹不要求处处负定由 −x⁴ 及其导数直接验证。
- [[01_Math/08_MathsCamp-EC400/01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#7.4 二阶判据：一点的曲率与处处的曲率|SOFP Lecture 1 对应讲解]]：保留完整推导、条件、例子及课程语境。

<!-- bilingual-en:start -->
- The slide supplies the ordinary Hessian criterion; the strict sufficient condition follows along segments, and the quartic directly disproves necessity.
- The linked course exposition retains the detailed reasoning, assumptions, examples, and context.
<!-- bilingual-en:end -->

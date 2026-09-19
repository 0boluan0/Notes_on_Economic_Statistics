---
student_os: knowledge-atom
atom_id: OPT-TOOLS-010
aliases:
  - "实二次型的 Hessian 等于其表示矩阵与转置之和"
  - "The Hessian of a real quadratic form is its representing matrix plus its transpose"
status: source-checked
---

# 实二次型的 Hessian 等于其表示矩阵与转置之和

<!-- bilingual-en:start -->
*The Hessian of a real quadratic form is its representing matrix plus its transpose*
<!-- bilingual-en:end -->

若 $Q(x)=x^TAx$，则 $\nabla Q(x)=(A+A^T)x$，$H_Q=A+A^T$。选[[二次型对称化|对称代表]]后有 $\nabla Q=2Ax,H_Q=2A$。

二维 $Q=ax_1^2+2bx_1x_2+cx_2^2$ 连续求导得到 $H_Q=\begin{pmatrix}2a&2b\\2b&2c\end{pmatrix}$。因此 Taylor 中 $\tfrac12h^TH_Qh=h^TAh$。增加一次项和常数项不改变 Hessian，但此时 $h^THh=2f(h)$ 一般不再成立。

<!-- bilingual-en:start -->
Differentiating a real quadratic form gives the symmetric sum of its representing matrix. For the symmetric representative this becomes twice the matrix, cancelling Taylor's factor one half. Linear and constant additions leave the Hessian unchanged but invalidate the special identity with twice the whole function value.
<!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/08_MathsCamp-EC400/2026_Course_Materials/02_SOFP/Lectures/EC400 Slides Lecture 1.pdf#page=41|EC400 SOFP 原材料，PDF p.41]]：课件给出 Hessian 与二阶 Taylor 公式；本卡对二次型逐项求导，核对 A+Aᵀ 及对称时的 2A。
- [[01_Math/08_MathsCamp-EC400/01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#4.8 二次型的 Hessian 为什么是两倍系数矩阵|SOFP Lecture 1 对应讲解]]：保留完整推导、条件、例子及课程语境。

<!-- bilingual-en:start -->
- The slide supplies the Hessian and Taylor notation; direct differentiation verifies the symmetric sum and the factor two.
- The linked course exposition retains the detailed reasoning, assumptions, examples, and context.
<!-- bilingual-en:end -->

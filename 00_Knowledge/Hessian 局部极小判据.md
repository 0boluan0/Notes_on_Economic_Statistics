---
aliases:
  - "在 $C^2$ 函数的驻点处，正定 Hessian 推出严格局部极小，但不自动推出全局极小"
  - "驻点处正定 Hessian 只保证严格局部极小而非自动全局极小"
  - Positive-definite Hessian test
  - Second-derivative test boundary
student_os: knowledge-atom
atom_id: LA-SPD-022
atom_set: symmetric-positive-definite
atom_type: sufficient-condition
status: source-checked
mastery_state: unassessed
requires:
  - "[[正定矩阵]]"
  - "[[多元Taylor近似]]"
related:
  - "[[正定二次型的椭球]]"
  - "[[半定Hessian无结论]]"
  - "[[凸优化全局性]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
  - "[[多元微分.canvas]]"
---

# 在 $C^2$ 函数的驻点处，正定 Hessian 推出严格局部极小，但不自动推出全局极小
<!-- bilingual-en:start -->
*A positive-definite Hessian at a stationary point guarantees a strict local minimum, not automatically a global one*
<!-- bilingual-en:end -->

> [!summary] 局部极小判据
> 设 $f$ 在 $x_*$ 的邻域内为 $C^2$，且 $\nabla f(x_*)=0$。若
> $$\nabla^2f(x_*)\succ0,$$
> 则 $x_*$ 是严格局部极小点。这个点上一次 Hessian 检查本身不证明全局极小。
> <!-- bilingual-en:start -->
> Let $f$ be $C^2$ near a stationary point $x_*$. If $\nabla^2f(x_*)$ is positive definite, then $x_*$ is a strict local minimizer. A Hessian check at this single point does not by itself establish global minimality.
> <!-- bilingual-en:end -->

Taylor 展开给出
$$
f(x_*+h)=f(x_*)+\frac12h^T\nabla^2f(x_*)h+o(\|h\|^2).
$$
正定二次项在所有小非零方向上为正，并支配余项，因此得到严格局部结论。
<!-- bilingual-en:start -->
The second-order Taylor expansion has leading change $\tfrac12h^T\nabla^2f(x_*)h$. Positive definiteness makes this term uniformly positive in every small nonzero direction, and it dominates the remainder.
<!-- bilingual-en:end -->

这条判据只控制 $x_*$ 附近。要把局部结论升级为全局结论，必须在整个凸域上控制曲率，见 [[凸优化全局性|凸优化的局部—全局关系]]；若 Hessian 在驻点只半正定，则二阶检验可能无结论，见 [[半定Hessian无结论]]。
<!-- bilingual-en:start -->
This criterion controls only a neighbourhood of $x_*$. A global conclusion requires curvature information across a convex domain; see [[凸优化全局性|the local-to-global result for convex optimisation]]. If the Hessian is only semidefinite at the stationary point, the second-order test may be inconclusive; see [[半定Hessian无结论|the inconclusive semidefinite-Hessian case]].
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么“$\nabla^2f(x_*)\succ0$”不能自动排除远处有更低的函数值？
>
> **答案：** 它只控制 $x_*$ 附近的二阶曲率，没有约束离开该邻域后的函数形状。
> <!-- bilingual-en:start -->
> Why does $\nabla^2f(x_*)\succ0$ not rule out a lower function value far from $x_*$?
>
> **Answer:** It controls second-order curvature only near $x_*$ and imposes no restriction on the shape of the function outside that neighbourhood.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.3sum.pdf|MIT 18.06SC Session 3.3 summary]]：核对正定二次型与极小值的联系。
- [MIT 18.S096, *Second Derivatives, Bilinear Maps, and Hessian Matrices*](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec12.pdf#page=2)：直接核对多元 Taylor 展开与 Hessian 二次项。
- [Boyd and Vandenberghe, *Convex Optimization*, §3.1.3–3.1.4](https://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=82)：核对凸域上处处 PSD Hessian 的全局凸性结论及凸函数驻点的全局最优性。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.3.3 Hessian 与极小值|课程 3.3.3]]：核对局部/全局边界与二次函数特例。
<!-- bilingual-en:start -->
- The MIT Session 3.3 summary was checked for the relation between positive-definite quadratic forms and minima.
- MIT 18.S096 notes directly support the multivariable Taylor expansion and its Hessian quadratic term.
- Boyd and Vandenberghe were checked for the everywhere-PSD Hessian criterion on a convex domain and the global optimality of a stationary point of a convex function.
- Course Section 3.3.3 was checked for the local/global boundary and the quadratic special case.
<!-- bilingual-en:end -->

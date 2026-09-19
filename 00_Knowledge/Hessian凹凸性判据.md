---
student_os: knowledge-atom
atom_id: OPT-TOOLS-015
aliases:
  - "开凸域上的二阶连续可微函数凹当且仅当 Hessian 处处半负定"
  - "A C2 function on an open convex domain is concave exactly when its Hessian is everywhere negative semidefinite"
status: needs-review
---

# 开凸域上的二阶连续可微函数凹当且仅当 Hessian 处处半负定

<!-- bilingual-en:start -->
*A C2 function on an open convex domain is concave exactly when its Hessian is everywhere negative semidefinite*
<!-- bilingual-en:end -->

若 $U$ 为开凸集、$f\in C^2(U)$，则 $f$ 凹当且仅当 $H_f(x)\preceq0$ 对每个 $x\in U$ 成立；凸对应处处半正定。矩阵的半定号要求所有方向的二次型符号，不是元素逐项符号。

沿任意线段令 $g(t)=f(x+t(y-x))$，由[[Hessian方向二阶导数]]把条件转为 $g''(t)\le0$，再用[[导数判凹凸]]；反向限制到每个点的局部直线即可。

一个点的 Hessian 定号不足以证明全域凹性。低维可行域应检验可行方向；含边界时需要额外的连续延拓论证。

<!-- bilingual-en:start -->
The sign condition must hold at every point and in every direction. Restricting to line segments uses [[Hessian方向二阶导数|the directional Hessian identity]] and [[导数判凹凸|the one-dimensional criterion]] to prove it. One-point curvature cannot establish global concavity. Lower-dimensional domains and included boundary points require appropriate directional or extension arguments.
<!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/08_MathsCamp-EC400/2026_Course_Materials/02_SOFP/Lectures/EC400 Slides Lecture 1.pdf#page=49|EC400 SOFP 原材料，PDF p.49]]：核对处处半负定与凹性的等价关系；正文明确开凸域和二阶连续可微条件，并用直线限制核对全方向要求。
- [[01_Math/08_MathsCamp-EC400/01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#7.4 二阶判据：一点的曲率与处处的曲率|SOFP Lecture 1 对应讲解]]：保留完整推导、条件、例子及课程语境。

<!-- bilingual-en:start -->
- The source states the Hessian criterion; the text makes the open convex domain and C2 assumptions explicit and checks all directions through line restrictions.
- The linked course exposition retains the detailed reasoning, assumptions, examples, and context.
<!-- bilingual-en:end -->

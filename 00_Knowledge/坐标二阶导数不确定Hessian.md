---
student_os: knowledge-atom
atom_id: OPT-TOOLS-011
aliases:
  - "只知道坐标方向二阶导数不能确定交叉曲率或完整 Hessian"
  - "Coordinate-direction second derivatives alone do not determine mixed curvature or the full Hessian"
status: needs-review
---

# 只知道坐标方向二阶导数不能确定交叉曲率或完整 Hessian

<!-- bilingual-en:start -->
*Coordinate-direction second derivatives alone do not determine mixed curvature or the full Hessian*
<!-- bilingual-en:end -->

对于对称 Hessian $H$，坐标方向的值 $e_i^THe_i=H_{ii}$ 只给对角元。$f(x,y)=xy$ 的这两个值都为零，但沿 $(1,1)$ 的二阶导数为 $2$，沿 $(1,-1)$ 为 $-2$。

令 $q(v)=v^THv$，则
$$
H_{ij}=\frac{q(e_i+e_j)-q(e_i)-q(e_j)}2.
$$
因此选择 $n$ 条坐标方向和 $n(n-1)/2$ 条成对和方向，可以恢复完整 Hessian；任意同样数量的方向未必足够。这与一阶导数对方向线性、由坐标值即可确定不同。见[[Hessian方向二阶导数]]。

<!-- bilingual-en:start -->
Coordinate directions reveal only Hessian diagonal entries. The product example hides opposite diagonal curvatures from those measurements. Adding pairwise sums of basis directions recovers mixed entries by the displayed identity. These specifically chosen measurements determine the Hessian; arbitrary measurements of the same count need not. See [[Hessian方向二阶导数|directional second derivatives]].
<!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/08_MathsCamp-EC400/2026_Course_Materials/02_SOFP/Lectures/EC400 Slides Lecture 1.pdf#page=41|EC400 SOFP 原材料，PDF p.41]]：课件给出完整 Hessian 的混合偏导项；本卡用 xy 和极化计算直接检验坐标测量的不足及补充方向。
- [[01_Math/08_MathsCamp-EC400/01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#4.7 再用一次链式法则，得到方向二阶导数|SOFP Lecture 1 对应讲解]]：保留完整推导、条件、例子及课程语境。

<!-- bilingual-en:start -->
- The slide includes mixed Hessian entries; the product example and polarization calculation check what coordinate measurements miss.
- The linked course exposition retains the detailed reasoning, assumptions, examples, and context.
<!-- bilingual-en:end -->

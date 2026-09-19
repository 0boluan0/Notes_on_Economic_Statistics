---
aliases:
  - "$C^2$ 函数驻点处的奇异半定 Hessian 不能单独判断局部极值类型"
  - "驻点处的奇异半定 Hessian 不能单独判断局部极值类型"
  - A singular semidefinite Hessian at a stationary point is generally inconclusive
student_os: knowledge-atom
atom_id: LA-SPD-029
atom_set: symmetric-positive-definite
atom_type: diagnostic-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[半正定矩阵]]"
  - "[[多元Taylor近似]]"
related:
  - "[[Hessian 局部极小判据]]"
  - "[[Hessian 局部极大判据]]"
  - "[[Hessian 鞍点判据]]"
  - "[[严格最优不推系统非奇异]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
  - "[[多元微分.canvas]]"
  - "[[多元优化.canvas]]"
---

# $C^2$ 函数驻点处的奇异半定 Hessian 不能单独判断局部极值类型
<!-- bilingual-en:start -->
*A singular semidefinite Hessian at a stationary point is not enough to determine the local-extremum type*
<!-- bilingual-en:end -->

> [!summary] 二阶检验的边界
> 设 $f$ 在驻点 $x_*$ 附近为 $C^2$。若 $\nabla^2f(x_*)$ 半正定但非正定，或半负定但非负定，二阶检验一般没有结论：Hessian 的零方向会让二次项消失，而未显示的高阶项可能把该方向弯向上、弯向下，或造成鞍点。
> <!-- bilingual-en:start -->
> At a $C^2$ stationary point, a singular positive- or negative-semidefinite Hessian is generally inconclusive. The quadratic term vanishes along its null directions, leaving higher-order terms free to create a minimum, a maximum, or a saddle.
> <!-- bilingual-en:end -->

两个函数
$$
f_+(x,y)=x^2+y^4,
\qquad
f_-(x,y)=x^2-y^4
$$
在原点都有
$$
\nabla f_\pm(0,0)=0,
\qquad
\nabla^2f_\pm(0,0)=\begin{bmatrix}2&0\\0&0\end{bmatrix}\succeq0.
$$
但 $f_+$ 在原点有严格局部极小，而 $f_-$ 沿 $x$ 轴为正、沿 $y$ 轴为负，所以原点是鞍点。同一个半正定 Hessian 对应两种不同局部行为，已经足以证明二阶信息不完备。
<!-- bilingual-en:start -->
The two displayed functions have the same positive-semidefinite Hessian $\operatorname{diag}(2,0)$ at the origin. The first has a strict local minimum; the second is positive along the $x$ axis and negative along the $y$ axis, so it has a saddle. Identical second-order data can therefore lead to different local behaviour.
<!-- bilingual-en:end -->

严格定号时情况不同：正定 Hessian 推出严格局部极小，负定推出严格局部极大，不定推出鞍点。奇异半定不是“几乎通过”的弱结论，而是明确告诉你：必须继续检查高阶项、函数的特殊结构，或邻域内的直接符号。
<!-- bilingual-en:start -->
Strict signs are decisive: positive definiteness gives a strict local minimum, negative definiteness a strict local maximum, and indefiniteness a saddle. A singular semidefinite Hessian is not an almost-complete verdict; it is a signal to inspect higher-order terms, special structure, or direct signs in a neighbourhood.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么 $f_-(x,y)=x^2-y^4$ 的半正定 Hessian 没有排除鞍点？
>
> **答案：** Hessian 在 $y$ 方向为零，负变化直到四阶项 $-y^4$ 才出现；二阶检验看不到它。
> <!-- bilingual-en:start -->
> Why does the positive-semidefinite Hessian of $f_-(x,y)=x^2-y^4$ fail to rule out a saddle point?
>
> **Answer:** The Hessian vanishes in the $y$ direction, while the negative change appears only in the fourth-order term $-y^4$. A second-order test cannot detect it.
> <!-- bilingual-en:end -->

这里采用宽义半定定义，所以必须明确“非定”或“奇异”：[[正定矩阵|正定]]本来也属于[[半正定矩阵|半正定]]，不在无结论的退化情况内。已完整识别一个奇异半定 Hessian 后，再计算它的其他主子式不能补回高阶信息；应该回到原函数。
<!-- bilingual-en:start -->
The inclusive definition of semidefiniteness requires the qualifier “singular” or “not definite”: [[正定矩阵|positive-definite matrices]] are also [[半正定矩阵|positive semidefinite]] and do not belong to the inconclusive case. Once a singular semidefinite Hessian is fully known, additional minors cannot recover higher-order terms; return to the original function.
<!-- bilingual-en:end -->

## 来源与核验

- [[多元Taylor近似]]：核对驻点处 Hessian 二次型与 $o(\|h\|^2)$ 余项的分工。
- [MIT 18.S096, *Second Derivatives, Bilinear Maps, and Hessian Matrices*](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec12.pdf#page=2)：核对二阶 Taylor 展开与 Hessian 判别的适用条件。
- LSE EC400，*SOFP Slides Lecture 2*（课程讲义）：核对课程中的正定、负定、不定与半定分类边界。
<!-- bilingual-en:start -->
- [[多元Taylor近似|The multivariable Taylor approximation]] was checked for the division of responsibility between the Hessian quadratic term and the higher-order remainder.
- MIT 18.S096 notes were checked for the second-order Taylor expansion and the scope of Hessian tests.
- EC400 SOFP Slides Lecture 2 were checked for the course classification of definite, indefinite, and semidefinite Hessians.
<!-- bilingual-en:end -->

- [[01_Math/08_MathsCamp-EC400/2026_Course_Materials/02_SOFP/Lectures/EC400 Lecture Notes SOFP.pdf#page=21|SOFP 讲义 p.21]]：核对半定不充分与正负定充分的区别；[[01_Math/08_MathsCamp-EC400/01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#6.3 半定无结论不是“主子式算得还不够多”|本课两种同 Hessian 的例子]]直接展示缺少的高阶信息。

<!-- bilingual-en:start -->
The linked SOFP notes distinguish semidefinite necessity from definite sufficiency. The course examples share the same Hessian but have different higher-order behaviour.
<!-- bilingual-en:end -->

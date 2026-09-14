---
aliases:
  - '二阶可微函数的局部变化由梯度线性项与 Hessian 二次项控制，剩余误差比扰动平方更小'
  - A second-order Taylor approximation combines the gradient term with the Hessian quadratic term
student_os: knowledge-atom
atom_id: CALC-MV-009
atom_set: multivariable-differentiation
atom_type: approximation-theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[全微分]]"
  - "[[Hessian矩阵]]"
leads_to:
  - "[[Hessian 局部极小判据]]"
  - "[[Hessian 局部极大判据]]"
  - "[[Hessian 鞍点判据]]"
  - "[[半定Hessian无结论]]"
part_of:
  - "[[多元微分.canvas|多元微分]]"
---

# 二阶可微函数的局部变化由梯度线性项与 Hessian 二次项控制，剩余误差比扰动平方更小
<!-- bilingual-en:start -->
*The local change of a twice-differentiable function is governed by a linear gradient term and a quadratic Hessian term, with an error smaller than the squared perturbation*
<!-- bilingual-en:end -->

> [!summary] Taylor 展开把局部变化按阶数分开
> 若 $f:\mathbb R^n\to\mathbb R$ 在 $a$ 的邻域为 $C^2$，则当 $h\to0$ 时
> $$
> f(a+h)=f(a)+\nabla f(a)^Th
> +\frac12h^T\nabla^2f(a)h+o(\|h\|^2).
> $$
> 第一项是基准值，第二项是一阶线性响应，第三项是二阶曲率修正；余项相对于 $\|h\|^2$ 趋于零。
>
> <!-- bilingual-en:start -->
> For a $C^2$ scalar function, the displayed expansion separates the baseline value, the first-order linear response, the second-order curvature correction, and a remainder negligible relative to $\|h\|^2$.
> <!-- bilingual-en:end -->

## 一阶与二阶近似回答不同精度的问题
<!-- bilingual-en:start -->
*First- and second-order approximations answer questions at different precision*
<!-- bilingual-en:end -->

只保留梯度项得到
$$
f(a+h)-f(a)=\nabla f(a)^Th+o(\|h\|).
$$
加入 Hessian 二次项后，已解释的误差尺度从一阶推进到二阶。若 $\nabla f(a)=0$，线性项消失，Hessian 二次型成为最先需要检查的候选主导项；若它也在某些方向为零，更高阶项可能决定局部形状。

<!-- bilingual-en:start -->
Keeping only the gradient yields the first-order expansion shown above. Adding the Hessian improves the controlled error scale from first to second order. At a stationary point the linear term vanishes, making the Hessian quadratic form the first candidate for the leading change; if that form is flat in some directions, higher-order terms may decide the local geometry.
<!-- bilingual-en:end -->

## 完整例子：线性项、二次项和真实误差
<!-- bilingual-en:start -->
*Worked example: the linear term, quadratic correction, and exact error*
<!-- bilingual-en:end -->

令 $f(x,y)=x^2y$，在 $a=(1,2)$ 取
$$
h=\begin{pmatrix}0.01\\-0.02\end{pmatrix}.
$$
此时
$$
\nabla f(a)=\begin{pmatrix}4\\1\end{pmatrix},
\qquad
\nabla^2f(a)=\begin{pmatrix}4&2\\2&0\end{pmatrix}.
$$
一阶项为
$$
\nabla f(a)^Th=0.02,
$$
二阶修正为
$$
\frac12h^T\nabla^2f(a)h=-0.0002.
$$
所以二阶近似预测增量 $0.0198$。直接计算得到
$$
f(1.01,1.98)-f(1,2)=0.019798,
$$
剩余误差为 $-0.000002$，正是该三次多项式尚未被二阶展开捕捉的 $dx^2dy$ 项。

<!-- bilingual-en:start -->
For $f(x,y)=x^2y$ at $(1,2)$, the displayed perturbation gives a first-order change of $0.02$ and a second-order correction of $-0.0002$. Their sum, $0.0198$, differs from the exact increment $0.019798$ by only $-0.000002$, the cubic term $dx^2dy$ omitted by the quadratic approximation.
<!-- bilingual-en:end -->

## 从 Taylor 到局部极值分类
<!-- bilingual-en:start -->
*From Taylor expansion to local-extremum classification*
<!-- bilingual-en:end -->

在驻点处，若 Hessian 正定，二次项在所有非零小方向为正，得到现有的 [[Hessian 局部极小判据]]；若负定，得到 [[Hessian 局部极大判据]]；若不定，则不同方向出现相反符号，得到 [[Hessian 鞍点判据]]。若只半正定或半负定，二次项可能沿非零方向消失，[[半定Hessian无结论|二阶检验通常没有结论]]。

<!-- bilingual-en:start -->
At a stationary point, a positive-definite Hessian yields the existing [[Hessian 局部极小判据|strict local-minimum criterion]], a negative-definite Hessian yields the [[Hessian 局部极大判据|strict local-maximum criterion]], and an indefinite Hessian yields the [[Hessian 鞍点判据|saddle criterion]]. A semidefinite Hessian may vanish along nonzero directions, making the [[半定Hessian无结论|second-order test inconclusive]].
<!-- bilingual-en:end -->

这些结论都只控制 $a$ 附近。Taylor 展开在一点的局部信息不能自动排除远处出现更高或更低的函数值；全局结论需要整个定义域上的凸性、凹性或直接比较。

<!-- bilingual-en:start -->
All of these conclusions are local. A Taylor expansion at one point cannot rule out larger or smaller values far away; global conclusions require convexity, concavity, or direct comparison over the whole feasible domain.
<!-- bilingual-en:end -->

> [!question]- 自检：梯度项消失后能否直接宣布极值？
> 若 $\nabla f(a)=0$，为什么还必须检查 Hessian 的二次型而不是只看“线性近似为零”？
>
> <!-- bilingual-en:start -->
> If $\nabla f(a)=0$, why must the Hessian quadratic form still be examined instead of concluding from the zero linear approximation?
> <!-- bilingual-en:end -->
>
> **答案：** 零梯度只消除了所有一阶变化。二阶项可能在不同方向同号、反号或为零，分别对应不同的局部行为；若二阶项退化，还要继续看更高阶结构。
>
> <!-- bilingual-en:start -->
> **Answer:** A zero gradient removes only first-order change. The quadratic term may have one sign in every direction, opposite signs in different directions, or vanish in some directions, leading to different local behaviour; a degenerate quadratic term may require higher-order analysis.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- LSE EC400，*Revision Maths Notes 7: Working with Multivariate Calculus*（课程讲义） §7.9：直接核对多元二次 Taylor 公式、向量—矩阵写法和沿直线化为一元展开的推导。
- [MIT 18.S096, Lecture Notes and Readings](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/pages/lecture-notes-and-readings/)：交叉核对 Hessian 二次项、二阶余项和局部曲率解释。

<!-- bilingual-en:start -->
- EC400 Revision Maths Notes 7, §7.9, was checked directly for the multivariable quadratic Taylor formula, its vector-matrix form, and its derivation by restricting the function to a line.
- MIT 18.S096 was used to cross-check the Hessian quadratic term, the second-order remainder, and its local-curvature interpretation.
<!-- bilingual-en:end -->

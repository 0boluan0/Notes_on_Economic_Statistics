---
aliases:
  - "Hessian 是实值函数全部二阶偏导排成的方阵；当梯度可微时，它就是梯度的 Jacobian"
  - The Hessian is the square matrix of all second partial derivatives of a real-valued function and is the Jacobian of its gradient when that gradient is differentiable
student_os: knowledge-atom
atom_id: CALC-MV-008
atom_set: multivariable-differentiation
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Jacobian矩阵]]"
leads_to:
  - "[[Hessian对称条件]]"
  - "[[Hessian方向二阶导数]]"
part_of:
  - "[[多元微分.canvas|多元微分]]"
---

# Hessian 是实值函数全部二阶偏导排成的方阵；当梯度可微时，它就是梯度的 Jacobian
<!-- bilingual-en:start -->
*The Hessian is the square matrix of all second partial derivatives of a real-valued function and is the Jacobian of its gradient when that gradient is differentiable*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 对实值函数 $f:\mathbb R^n\to\mathbb R$，若所需二阶偏导在 $a$ 存在，则 Hessian 是 $n\times n$ 方阵
> $$
> \nabla^2f(a)=\left[\frac{\partial}{\partial x_j}
> \left(\frac{\partial f}{\partial x_i}\right)(a)\right]_{i,j=1}^n.
> $$
> 对角元记录同一坐标上的二次求导，非对角元记录两个坐标之间的交叉偏导。若梯度映射在 $a$ 可微，则
> $$
> \nabla^2f(a)=J_{\nabla f}(a).
> $$
> <!-- bilingual-en:start -->
> For a scalar-valued function $f:\mathbb R^n\to\mathbb R$, the Hessian at $a$ is the displayed $n\times n$ matrix of second partial derivatives whenever those entries exist. Diagonal entries differentiate twice with respect to the same coordinate, while off-diagonal entries are mixed partials. If the gradient map is differentiable at $a$, the Hessian is precisely its Jacobian.
> <!-- bilingual-en:end -->

## 为什么它是梯度的 Jacobian
<!-- bilingual-en:start -->
*Why it is the Jacobian of the gradient*
<!-- bilingual-en:end -->

梯度是向量值映射
$$
\nabla f(x)=
\begin{pmatrix}
\partial_1f(x)\\
\vdots\\
\partial_nf(x)
\end{pmatrix}.
$$
对它逐分量求导后，Jacobian 的第 $(i,j)$ 个条目就是 $\partial_j(\partial_i f)$，恰好是 Hessian 的第 $(i,j)$ 个条目。仅凭这些二阶偏导存在，不能自动断言矩阵对称；交叉偏导何时相等由 [[Hessian对称条件]] 给出。
<!-- bilingual-en:start -->
The gradient is the displayed vector-valued map. Differentiating it component by component makes entry $(i,j)$ of its Jacobian equal to $\partial_j(\partial_i f)$, exactly entry $(i,j)$ of the Hessian. The mere existence of these second partials does not by itself make the matrix symmetric; [[Hessian对称条件|the Hessian symmetry condition]] states when the mixed partials agree.
<!-- bilingual-en:end -->

## 交叉偏导记录坐标之间的局部作用
<!-- bilingual-en:start -->
*Mixed partials record local interactions between coordinates*
<!-- bilingual-en:end -->

对 $f(x,y)=x^2y$，
$$
\nabla f(x,y)=
\begin{pmatrix}2xy\\x^2\end{pmatrix},
\qquad
\nabla^2f(x,y)=
\begin{pmatrix}2y&2x\\2x&0\end{pmatrix}.
$$
条目 $\partial_y(\partial_xf)=2x$ 表示 $y$ 改变时，$x$ 方向的一阶变化率怎样改变；条目 $\partial_x(\partial_yf)=2x$ 从另一顺序记录同一对坐标的交叉作用。这个多项式足够光滑，所以两者相等。
<!-- bilingual-en:start -->
For $f(x,y)=x^2y$, the displayed Hessian has mixed entries equal to $2x$. The entry $\partial_y(\partial_xf)$ measures how the first-order rate in the $x$ direction changes with $y$, while $\partial_x(\partial_yf)$ records the interaction in the opposite order. The polynomial is smooth enough for these two entries to agree.
<!-- bilingual-en:end -->

Hessian 如何把这些条目组合成沿任意方向的二阶变化，是另一个可单独使用的恒等式，见 [[Hessian方向二阶导数]]。
<!-- bilingual-en:start -->
How the Hessian combines these entries into second-order change along an arbitrary direction is a separate reusable identity; see [[Hessian方向二阶导数|the Hessian formula for second directional derivatives]].
<!-- bilingual-en:end -->

## 向量值函数的二阶导数不再是一张 Hessian
<!-- bilingual-en:start -->
*A vector-valued second derivative is not one Hessian matrix*
<!-- bilingual-en:end -->

Hessian 矩阵直接描述实值函数。若 $F:\mathbb R^n\to\mathbb R^m$ 且 $m>1$，每个分量 $F_k$ 可以有自己的 $n\times n$ Hessian；整体二阶导数更自然地看作一个双线性映射，或在坐标中表示为三阶数组。单独一张 $n\times n$ 矩阵不能容纳全部向量输出的二阶信息。
<!-- bilingual-en:start -->
The Hessian matrix directly describes a scalar-valued function. For $F:\mathbb R^n\to\mathbb R^m$ with $m>1$, each component $F_k$ can have its own $n\times n$ Hessian. The full second derivative is more naturally a bilinear map, or a third-order array in coordinates. A single $n\times n$ matrix cannot contain all second-order information for a vector output.
<!-- bilingual-en:end -->

> [!question]- 自检
> 若 $f:\mathbb R^3\to\mathbb R$，它的 Hessian 是什么尺寸，第 $(2,3)$ 个条目表示什么？若改为 $F:\mathbb R^3\to\mathbb R^2$，为什么一张 $3\times3$ 矩阵不再足够？
> <!-- bilingual-en:start -->
> If $f:\mathbb R^3\to\mathbb R$, what is the size of its Hessian and what does entry $(2,3)$ mean? Why is one $3\times3$ matrix insufficient for $F:\mathbb R^3\to\mathbb R^2$?
> <!-- bilingual-en:end -->
>
> **答案：** Hessian 是 $3\times3$，第 $(2,3)$ 个条目是 $\partial_3(\partial_2f)$。向量值映射有两个标量分量，每个分量各有一张 Hessian；整体二阶导数因此需要两张矩阵或等价的三阶表示。
> <!-- bilingual-en:start -->
> **Answer:** The Hessian is $3\times3$, and entry $(2,3)$ is $\partial_3(\partial_2f)$. A map with two scalar components has one Hessian per component, so its full second derivative requires two matrices or an equivalent third-order representation.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- LSE EC400，*Revision Maths Notes 7: Working with Multivariate Calculus*（课程讲义） §7.7：核对二阶偏导、交叉偏导和 Hessian 的课程定义。
  <!-- bilingual-en:start -->
  *English:* EC400 Revision Maths Notes 7, Section 7.7, supports the course definitions of second partial derivatives, mixed partials, and the Hessian matrix.
  <!-- bilingual-en:end -->
- [MIT 18.S096, *Second Derivatives, Bilinear Maps, and Hessian Matrices*](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec12.pdf)：核对 Hessian 作为梯度的 Jacobian，以及标量值与向量值二阶导数的表示差别。
  <!-- bilingual-en:start -->
  *English:* MIT 18.S096 notes support the Hessian as the Jacobian of the gradient and the distinction between scalar-valued Hessian matrices and vector-valued second derivatives.
  <!-- bilingual-en:end -->

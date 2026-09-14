---
aliases:
  - '若 $f$ 在 $a$ 附近二阶连续可微，则沿 $v$ 的二阶导数等于 $v^T\nabla^2f(a)v$'
  - The second directional derivative of a C2 function is the Hessian quadratic form
student_os: knowledge-atom
atom_id: CALC-MV-023
atom_set: multivariable-differentiation
atom_type: identity
status: source-checked
mastery_state: unassessed
requires:
  - "[[Hessian矩阵]]"
  - "[[多元链式法则]]"
related:
  - "[[方向导数梯度公式]]"
leads_to:
  - "[[多元Taylor近似]]"
part_of:
  - "[[多元微分.canvas|多元微分]]"
---

# 若 $f$ 在 $a$ 附近二阶连续可微，则沿 $v$ 的二阶导数等于 $v^T\nabla^2f(a)v$
<!-- bilingual-en:start -->
*If $f$ is twice continuously differentiable near $a$, its second derivative along $v$ is $v^T\nabla^2f(a)v$*
<!-- bilingual-en:end -->

> [!summary] 核心恒等式
> 固定 $v\in\mathbb R^n$，并令
> $$
> \phi(t)=f(a+tv).
> $$
> 若 $f$ 在 $a$ 附近为 $C^2$，则
> $$
> \phi''(0)=v^T\nabla^2f(a)v.
> $$
> Hessian 的二次型把全部纯二阶偏导和交叉偏导组合成沿这条直线观察到的二阶变化。
> <!-- bilingual-en:start -->
> Fix $v\in\mathbb R^n$ and define $\phi(t)=f(a+tv)$. If $f$ is $C^2$ near $a$, then $\phi''(0)=v^T\nabla^2f(a)v$. The Hessian quadratic form combines all pure and mixed second partial derivatives into the second-order change observed along this line.
> <!-- bilingual-en:end -->

## 推导
<!-- bilingual-en:start -->
*Derivation*
<!-- bilingual-en:end -->

对 $\phi$ 使用一次链式法则，
$$
\phi'(t)=\nabla f(a+tv)^Tv.
$$
再次求导时，$v$ 是固定向量，而梯度的 Jacobian 是 Hessian，因此
$$
\phi''(t)=v^T\nabla^2f(a+tv)v.
$$
令 $t=0$ 就得到所述恒等式。它说明 Hessian 不是若干互不相干的二阶偏导表格，而是一个把方向映成二阶变化率的二次型。
<!-- bilingual-en:start -->
One application of the chain rule gives $\phi'(t)=\nabla f(a+tv)^Tv$. Differentiating again, with $v$ fixed and the Jacobian of the gradient equal to the Hessian, yields $\phi''(t)=v^T\nabla^2f(a+tv)v$. Setting $t=0$ proves the identity. The Hessian is therefore not merely a table of unrelated second partial derivatives; it is a quadratic form that assigns a second-order rate of change to each direction.
<!-- bilingual-en:end -->

## 向量长度也会进入二阶变化率
<!-- bilingual-en:start -->
*The length of the direction vector also affects the second-order rate*
<!-- bilingual-en:end -->

把方向换成 $cv$ 后，
$$
(cv)^T\nabla^2f(a)(cv)=c^2v^T\nabla^2f(a)v.
$$
所以若要比较纯粹的几何方向，通常把 $v$ 归一化；若 $v$ 表示真实速度或有限参数化，则不应擅自归一化。
<!-- bilingual-en:start -->
Replacing $v$ by $cv$ multiplies the quadratic form by $c^2$. To compare geometric directions alone, one therefore usually normalizes $v$. If $v$ represents an actual velocity or parameterization, it should not be normalized automatically.
<!-- bilingual-en:end -->

## 最小例子
<!-- bilingual-en:start -->
*Minimal example*
<!-- bilingual-en:end -->

令 $f(x,y)=x^2+xy+y^2$，并取基点 $a=(0,0)$，则
$$
\nabla^2f=
\begin{bmatrix}2&1\\1&2\end{bmatrix}.
$$
沿 $v=(1,2)^T$，有 $f(t,2t)=7t^2$，故 $\phi''(0)=14$；矩阵计算也给出
$$
v^T\nabla^2f\,v=14.
$$
<!-- bilingual-en:start -->
For $f(x,y)=x^2+xy+y^2$ at the base point $a=(0,0)$, the Hessian is $\begin{bmatrix}2&1\\1&2\end{bmatrix}$. Along $v=(1,2)^T$, one has $f(t,2t)=7t^2$, so $\phi''(0)=14$; the quadratic form gives the same value.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> 为什么沿 $2v$ 的二阶导数是沿 $v$ 的四倍，而不是两倍？
> <!-- bilingual-en:start -->
> Why is the second derivative along $2v$ four times, rather than twice, the second derivative along $v$?
> <!-- bilingual-en:end -->
>
> **答案：** 方向向量在二次型的左右各出现一次，所以缩放因子变成 $2^2$。
> <!-- bilingual-en:start -->
> **Answer:** The direction vector appears once on each side of the quadratic form, so its scale factor is squared.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- LSE EC400，*Revision Maths Notes 7: Working with Multivariate Calculus*（课程讲义） §7.7：核验 Hessian、交叉偏导与二阶局部变化的课程定义。
- [MIT 18.S096, *Second Derivatives, Bilinear Maps, and Hessian Matrices*](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec12.pdf)：核验 Hessian 二次型、方向二阶导数与 Taylor 二次项的关系。
<!-- bilingual-en:start -->
- EC400 Revision Maths Notes 7, Section 7.7, supports the course definition of the Hessian, mixed partial derivatives, and second-order local change.
- MIT 18.S096 notes support the relation among the Hessian quadratic form, the second derivative along a direction, and the quadratic Taylor term.
<!-- bilingual-en:end -->

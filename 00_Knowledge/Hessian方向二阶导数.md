---
aliases:
  - Second directional derivative
  - Hessian quadratic-form identity
student_os: knowledge-atom
atom_id: CALC-MV-023
atom_set: multivariable-differentiation
atom_type: identity
status: source-checked
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

<!-- bilingual-en:start -->
# Second derivatives along a line
<!-- bilingual-en:end -->

*Hessian 方向二阶导数*

<!-- bilingual-en:start -->
A first derivative tells us the rate at which a quantity changes. A **second derivative along a line** tells us how that rate changes while the input keeps moving in one fixed direction at one fixed speed. For a function of several inputs, the Hessian combines all coordinate interactions into this one-dimensional answer.
<!-- bilingual-en:end -->

一阶导数告诉我们一个量变化得有多快。**沿直线的二阶导数**则描述：输入保持同一个方向、同一种速度移动时，这个变化率本身怎样改变。对于多个输入的函数，Hessian 把各坐标之间的作用组合成这个一元问题的答案。

<!-- bilingual-en:start -->
For example, take $f(x,y)=x^2+xy+y^2$ and move from the origin by setting $(x,y)=(t,2t)$. The observed value is $7t^2$, whose rate is $14t$ and whose second derivative is $14$. The mixed term $xy$ contributes to that answer because both coordinates move together.
<!-- bilingual-en:end -->

例如，对 $f(x,y)=x^2+xy+y^2$，从原点出发，令 $(x,y)=(t,2t)$。观察到的值是 $7t^2$，变化率为 $14t$，二阶导数为 $14$。两个坐标一起变化，所以交叉项 $xy$ 也会贡献这个结果。

<!-- bilingual-en:start -->
Let $U$ be an [[开集|open subset]] of $\mathbb R^n$, where $n\ge1$ is an integer, and let $f:U\to\mathbb R$. Choose a base point $a\in U$ and a fixed real column vector $v\in\mathbb R^n$. The input $a+tv$ moves along a line as the real parameter $t$ varies. Openness ensures that both positive and negative sufficiently small $t$ are allowed. Define the one-variable function $\phi(t)=f(a+tv)$.
<!-- bilingual-en:end -->

设 $U$ 是 $\mathbb R^n$ 中的[[开集]]，$n\ge1$ 为整数，$f:U\to\mathbb R$。选基点 $a\in U$，并固定实列向量 $v\in\mathbb R^n$。实参数 $t$ 改变时，输入 $a+tv$ 沿一条直线移动。开集条件保证足够小的正、负 $t$ 都合法。定义一元函数 $\phi(t)=f(a+tv)$。

> [!important] A sufficient smoothness condition · 一条充分的光滑性条件
> <!-- bilingual-en:start -->
> Assume $f$ is **$C^2$ near $a$**: its partial derivatives through order two exist and are continuous on an open neighborhood of $a$. Write $H_f(a)=\nabla^2f(a)$ for the [[Hessian矩阵|Hessian matrix]], the array of second partials. Then
>
> $$\phi''(0)=v^TH_f(a)v.$$
>
> Here $T$ means [[矩阵转置|transpose]]. The expression on the right is a [[二次型|quadratic form]]: multiply the $n\times n$ matrix by the $n$-component column, then by the row $v^T$, obtaining a scalar.
> <!-- bilingual-en:end -->
>
> 假设 $f$ **在 $a$ 附近为 $C^2$**：直到二阶的偏导都在 $a$ 的某个开邻域内存在且连续。用 $H_f(a)=\nabla^2f(a)$ 表示由二阶偏导组成的[[Hessian矩阵]]，则有上述恒等式。上标 $T$ 表示[[矩阵转置|转置]]；右侧是一个[[二次型]]，先将 $n\times n$ 矩阵乘以 $n$ 分量列向量，再左乘行向量 $v^T$，最后得到一个标量。

<!-- bilingual-en:start -->
The vector need not have length one. If $v=0$, the input stays at $a$, so $\phi$ is constant and both sides are zero. The $C^2$ assumption is a convenient sufficient hypothesis; the mere existence of a table of second partial derivatives at $a$ does not establish the identity.
<!-- bilingual-en:end -->

向量不必为单位长度。若 $v=0$，输入一直停在 $a$，因此 $\phi$ 是常函数，两侧都为零。$C^2$ 是方便检查的充分条件；仅有 $a$ 处的一张二阶偏导表，还不足以建立这条恒等式。

<!-- bilingual-en:start -->
## Apply the chain rule twice
<!-- bilingual-en:end -->

## 推导

<!-- bilingual-en:start -->
The [[全微分#实值函数、梯度与全微分|gradient]] $\nabla f$ is the column of first partial derivatives. The [[多元链式法则|chain rule]] gives, for every sufficiently small $t$,
<!-- bilingual-en:end -->

[[全微分#实值函数、梯度与全微分|梯度]] $\nabla f$ 是一阶偏导排成的列向量。由[[多元链式法则]]，对每个足够小的 $t$，有

$$\phi'(t)=\nabla f(a+tv)^Tv.$$

<!-- bilingual-en:start -->
Now differentiate this scalar again. The vector $v$ stays fixed, while the gradient changes with the input. Under $C^2$, the gradient is differentiable and its [[Jacobian矩阵|Jacobian]] is $H_f$. Consequently
<!-- bilingual-en:end -->

再对这个标量求导。向量 $v$ 保持固定，梯度则随输入改变。在 $C^2$ 条件下，梯度映射可微，其 [[Jacobian矩阵]]就是 $H_f$，所以

$$\phi''(t)=\bigl(H_f(a+tv)v\bigr)^Tv=v^TH_f(a+tv)v.$$

<!-- bilingual-en:start -->
The last equality holds because a scalar equals its transpose. Taking $t=0$ proves the result. This is a second derivative with respect to the parameter $t$; the [[方向导数梯度公式|first directional derivative formula]] is the preceding first-order step.
<!-- bilingual-en:end -->

最后一个等号使用标量等于自身转置。令 $t=0$ 即得结论。这里求的是关于参数 $t$ 的二阶导数；[[方向导数梯度公式]]是此前的一阶步骤。

<!-- bilingual-en:start -->
## Keep both mixed contributions
<!-- bilingual-en:end -->

## 二维展开中两个交叉项都来自链式法则

<!-- bilingual-en:start -->
In two dimensions write $f_i=\partial_i f$ and $f_{ij}=\partial_j(\partial_i f)$: differentiate in input $i$ first, then in input $j$. Holding $v_1,v_2$ fixed gives
<!-- bilingual-en:end -->

二维时记 $f_i=\partial_i f$、$f_{ij}=\partial_j(\partial_i f)$，即先对输入 $i$ 求导，再对输入 $j$ 求导。固定 $v_1,v_2$，得到

$$\begin{aligned}\phi''(t)&=(f_{11}v_1+f_{12}v_2)v_1+(f_{21}v_1+f_{22}v_2)v_2\\&=f_{11}v_1^2+(f_{12}+f_{21})v_1v_2+f_{22}v_2^2.\end{aligned}$$

<!-- bilingual-en:start -->
Every partial on the right is evaluated at $a+tv$. The [[Hessian对称条件|mixed-partial theorem]] justifies $f_{12}=f_{21}$ under $C^2$, giving $2f_{12}v_1v_2$. Along a coordinate axis that mixed contribution disappears; axis measurements therefore miss information needed for other directions. See [[坐标二阶导数不确定Hessian|what coordinate second derivatives do not determine]].
<!-- bilingual-en:end -->

右侧的偏导都在 $a+tv$ 取值。[[Hessian对称条件|混合偏导定理]]在 $C^2$ 下保证 $f_{12}=f_{21}$，从而交叉项合并为 $2f_{12}v_1v_2$。沿坐标轴时，交叉贡献会消失，因此轴上的测量会遗漏其他方向所需的信息，见[[坐标二阶导数不确定Hessian]]。

<!-- bilingual-en:start -->
## Rescaling the same line changes the parameter rate
<!-- bilingual-en:end -->

## 向量长度也会进入二阶变化率

$$(cv)^TH_f(a)(cv)=c^2v^TH_f(a)v\qquad(c\in\mathbb R).$$

<!-- bilingual-en:start -->
The vector occurs twice, so doubling it multiplies the second derivative by four. Reversing it leaves that second derivative unchanged. For comparison per unit distance, use $u=v/\|v\|_2$ when $v\ne0$, where $\|v\|_2$ is [[欧氏范数|Euclidean length]]. For an input with a specified physical speed or parameter scale, retain its actual vector. Normalizing it answers a different rate question.
<!-- bilingual-en:end -->

向量出现两次，所以把它加倍会让二阶导数变成四倍；把它反向则不改变这个二阶导数。若要按单位距离比较，对 $v\ne0$ 使用 $u=v/\|v\|_2$，其中 $\|v\|_2$ 是[[欧氏范数|欧氏长度]]。若输入已有指定的物理速度或参数尺度，应保留实际向量；归一化会改变所问的变化率。

<!-- bilingual-en:start -->
## Check the matrix against direct substitution
<!-- bilingual-en:end -->

## 最小例子

<!-- bilingual-en:start -->
Return to $f(x,y)=x^2+xy+y^2$, $a=(0,0)$ and $v=(1,2)^T$. Differentiation and multiplication give
<!-- bilingual-en:end -->

回到 $f(x,y)=x^2+xy+y^2$，取 $a=(0,0)$、$v=(1,2)^T$。求导并相乘，得到

$$H_f=\begin{pmatrix}2&1\\1&2\end{pmatrix},\qquad H_fv=\begin{pmatrix}4\\5\end{pmatrix},\qquad v^TH_fv=1(4)+2(5)=14.$$

<!-- bilingual-en:start -->
This agrees with $\phi(t)=7t^2$ and $\phi''(0)=14$. For $u=v/\sqrt5$, the answer is $14/5$; for $2v$ it is $56$. The coefficient of $t^2$ is $7$, not $14$: a [[多元Taylor近似|Taylor quadratic correction]] includes the factor $1/2$.
<!-- bilingual-en:end -->

这与 $\phi(t)=7t^2$、$\phi''(0)=14$ 一致。改用 $u=v/\sqrt5$ 时，答案为 $14/5$；改用 $2v$ 时为 $56$。注意 $t^2$ 的系数是 $7$，不是 $14$：[[多元Taylor近似|Taylor 二次修正]]还要乘上 $1/2$。

> [!warning] A changing velocity adds a term · 速度改变时还多一项
> <!-- bilingual-en:start -->
> Let $\gamma$ be a twice continuously differentiable path with $\gamma(0)=a$. For sufficiently small $t$, its image stays in the $C^2$ neighborhood of $f$, and the chain rule gives
>
> $$\frac{d^2}{dt^2}f(\gamma(t))=\gamma^{\prime}(t)^TH_f(\gamma(t))\gamma^{\prime}(t)+\nabla f(\gamma(t))^T\gamma^{\prime\prime}(t).$$
>
> For $f(x,y)=x$ and $\gamma(t)=(t^2,0)$, the Hessian is zero, but $f(\gamma(t))=t^2$ has second derivative $2$. The extra term comes from the changing input velocity. For $\gamma(t)=a+tv$, its second derivative vanishes, which is why the line formula has only one term.
> <!-- bilingual-en:end -->
>
> 设 $\gamma$ 是二阶连续可微路径且 $\gamma(0)=a$。对充分小的 $t$，路径留在 $f$ 的 $C^2$ 邻域内，链式法则给出的二阶导数还包含上述第二项。例如 $f(x,y)=x$、$\gamma(t)=(t^2,0)$ 时，Hessian 为零，但 $f(\gamma(t))=t^2$ 的二阶导数为 $2$；额外贡献来自输入速度的变化。直线参数化 $\gamma(t)=a+tv$ 的二阶导数为零，所以沿线公式只剩一项。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 18.S096, Second Derivatives, Bilinear Maps, and Hessian Matrices, §12.1, PDF pp.1–2 / print 87–88](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec12.pdf#page=1): Hessian as the gradient Jacobian, quadratic forms and the factor one half. The line and curved-path formulas above are derived by the chain rule with explicit regularity.
- [Lebl, Basic Analysis II, v6.3, §8.6, PDF pp.60–62](https://www.jirka.org/ra/realanal2.pdf#page=60): $C^2$, derivative order and mixed-partial symmetry.
- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/01_Revision_Mathematics/Notes/By_Topic/Revision Maths Notes 7, Working with Multivariate Calculus.pdf#page=11|LSE EC400 Notes 7, §§7.7 and 7.9.2, PDF pp.11 and 14]]: course notation and the two applications of the chain rule.
- [[EC400 Slides Lecture 1.pdf#page=41|EC400 SOFP Lecture 1, PDF p.41]] and [[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#4.7 再用一次链式法则，得到方向二阶导数|permanent course exposition §4.7]]: the quadratic-form context and the full mixed-term calculation; the examples here are checked directly.
<!-- bilingual-en:end -->

- MIT 18.S096 §12.1，PDF1–2页／印本87–88页：Hessian 是梯度的 Jacobian、二次型及系数二分之一。本文在明确正则条件后，用链式法则推导直线与曲线路径公式。
- Lebl《Basic Analysis II》§8.6，PDF60–62页：$C^2$、求导次序及混合偏导对称性。
- LSE EC400 Notes 7 §7.7、§7.9.2，PDF11、14页：课程记号以及两次应用链式法则。
- [[EC400 Slides Lecture 1.pdf#page=41|EC400 SOFP Lecture 1 原材料]]与[[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#4.7 再用一次链式法则，得到方向二阶导数|永久课程展开 §4.7]]：二次型语境与完整交叉项计算；本文算例另行直接核算。

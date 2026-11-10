---
aliases:
  - "Hessian symmetry"
  - "Clairaut–Schwarz theorem"
  - "Young定理"
student_os: knowledge-atom
atom_id: CALC-MV-016
atom_set: multivariable-differentiation
atom_type: theorem-boundary
status: source-checked
requires:
  - "[[Hessian矩阵]]"
related:
  - "[[实对称矩阵]]"
  - "[[二次型对称化]]"
leads_to:
  - "[[多元Taylor近似]]"
part_of:
  - "[[多元微分.canvas|多元微分]]"
---

<!-- bilingual-en:start -->
# Hessian symmetry
<!-- bilingual-en:end -->

*Hessian 对称条件*

<!-- bilingual-en:start -->
When two inputs interact, we can ask either how the $x$-rate changes with $y$, or how the $y$-rate changes with $x$. For $f(x,y)=x^2y$, both answers are $2x$. **Hessian symmetry** says that these two orders give the same answer for every pair of inputs under suitable regularity conditions. The condition matters: both answers can exist and still differ.
<!-- bilingual-en:end -->

两个输入发生交叉作用时，可以问“$x$ 方向变化率怎样随 $y$ 改变”，也可以问“$y$ 方向变化率怎样随 $x$ 改变”。对 $f(x,y)=x^2y$，两个答案都是 $2x$。**Hessian 对称性**说明：在适当的正则条件下，每一对输入的两种求导顺序都会得到同一答案。条件不能省，因为两个答案都存在，仍然可能不同。

<!-- bilingual-en:start -->
The [[Hessian矩阵|Hessian matrix]] uses $H_{ij}=\partial_j(\partial_i f)$: first take a [[偏导数|partial derivative]] in input $i$, then differentiate that result in input $j$. In two variables we write $f_{xy}=\partial_y(\partial_xf)$ and $f_{yx}=\partial_x(\partial_yf)$. A [[实对称矩阵|real symmetric matrix]] has equal reflected entries, $H_{ij}=H_{ji}$, or $H=H^T$, where $T$ means [[矩阵转置|transpose]].
<!-- bilingual-en:end -->

[[Hessian矩阵]]采用 $H_{ij}=\partial_j(\partial_i f)$：先对输入 $i$ 求[[偏导数]]，再对所得结果关于输入 $j$ 求导。两个变量时，记 $f_{xy}=\partial_y(\partial_xf)$、$f_{yx}=\partial_x(\partial_yf)$。[[实对称矩阵]]要求镜像位置的元素相同，即 $H_{ij}=H_{ji}$，也就是 $H=H^T$；$T$ 表示[[矩阵转置|转置]]。

> [!important] A standard sufficient condition · 常用的充分条件
> <!-- bilingual-en:start -->
> Let $f:U\to\mathbb R$, where $U\subseteq\mathbb R^n$ is [[开集|open]], $n\ge1$, and $a\in U$. Suppose all partial derivatives through order two exist and are continuous on an open neighborhood of $a$. This is the meaning of **$f$ being $C^2$ near $a$**. Then
> <!-- bilingual-en:end -->
>
> 设 $f:U\to\mathbb R$，其中 $U\subseteq\mathbb R^n$ 是[[开集]]，$n\ge1$，$a\in U$。假设直到二阶的所有偏导都在 $a$ 的某个开邻域内存在且连续，这就是 **$f$ 在 $a$ 附近为 $C^2$** 的含义。那么
>
> $$\partial_j(\partial_i f)(a)=\partial_i(\partial_j f)(a)
> \quad\text{for every }i,j,\qquad H_f(a)=H_f(a)^T.$$
>
> <!-- bilingual-en:start -->
> This is the usual Clairaut–Schwarz theorem, also called Young's theorem in the course notes. Continuity here concerns the multivariable input approaching a point from all nearby directions. The statement supplies a sufficient condition; it does not say that continuous second partials are necessary whenever two mixed partials happen to agree.
> <!-- bilingual-en:end -->
>
> 这就是通常的 Clairaut–Schwarz 定理，课程讲义也称 Young 定理。这里的连续性是多元输入从附近各种方向趋近一点时的连续性。这是一条充分条件，不是说只要两个混合偏导相等，就一定具有连续二阶偏导。

<!-- bilingual-en:start -->
## Why a small rectangle proves the result
<!-- bilingual-en:end -->

*用一个小矩形证明两种顺序相同*

<!-- bilingual-en:start -->
We prove the two-variable case at $(a,b)$. Choose positive $h,k$ small enough that the whole rectangle $[a,a+h]\times[b,b+k]$ stays in the neighborhood. Its four-corner difference is
<!-- bilingual-en:end -->

在 $(a,b)$ 处证明二元情形。取足够小的正数 $h,k$，使整个矩形 $[a,a+h]\times[b,b+k]$ 都在所给邻域内。考察四个角的函数值之差：

$$\Delta=f(a+h,b+k)-f(a,b+k)-f(a+h,b)+f(a,b).$$

<!-- bilingual-en:start -->
First view this as the change of $x\mapsto f(x,b+k)-f(x,b)$ from $a$ to $a+h$. The one-variable [[拉格朗日中值定理|mean value theorem]] gives an intermediate $\xi\in(a,a+h)$, and applying it again to $y\mapsto f_x(\xi,y)$ gives $\eta\in(b,b+k)$ such that
<!-- bilingual-en:end -->

先把它看成一元函数 $x\mapsto f(x,b+k)-f(x,b)$ 从 $a$ 到 $a+h$ 的变化。由[[拉格朗日中值定理]]，存在中间点 $\xi\in(a,a+h)$；再对 $y\mapsto f_x(\xi,y)$ 使用中值定理，得到 $\eta\in(b,b+k)$，使

$$\Delta=h\bigl[f_x(\xi,b+k)-f_x(\xi,b)\bigr]
=hk\,f_{xy}(\xi,\eta).$$

<!-- bilingual-en:start -->
Reversing the two roles gives possibly different intermediate points $\xi'\in(a,a+h)$ and $\eta'\in(b,b+k)$ with $\Delta=hk\,f_{yx}(\xi',\eta')$. Consequently
<!-- bilingual-en:end -->

交换两种输入的角色，同样得到可能不同的中间点 $\xi'\in(a,a+h)$、$\eta'\in(b,b+k)$，满足 $\Delta=hk\,f_{yx}(\xi',\eta')$。因此

$$\frac{\Delta}{hk}=f_{xy}(\xi,\eta)=f_{yx}(\xi',\eta').$$

<!-- bilingual-en:start -->
As $h,k\downarrow0$, both intermediate points approach $(a,b)$. Continuity of the two mixed partials makes the equal expressions tend to $f_{xy}(a,b)$ and $f_{yx}(a,b)$, proving equality. The $C^2$ assumption also supplies the continuity and differentiability needed for both applications of the mean value theorem. For $n$ inputs, fix all but the chosen pair and use the same small rectangle. The diagonal case $i=j$ is automatic.
<!-- bilingual-en:end -->

当 $h,k\downarrow0$ 时，两个中间点都趋于 $(a,b)$。两个混合偏导的连续性，使这些相等的表达式分别趋于 $f_{xy}(a,b)$ 和 $f_{yx}(a,b)$，从而证明相等。$C^2$ 前提也保证了两次使用中值定理所需的连续性与可导性。对 $n$ 个输入，固定其余坐标，在选定的一对坐标上重复这个小矩形论证即可；对角情形 $i=j$ 本来就相同。

<!-- bilingual-en:start -->
## Why existence at a point is not enough
<!-- bilingual-en:end -->

## 为什么“在一点存在”不够

<!-- bilingual-en:start -->
Consider the real-valued function
<!-- bilingual-en:end -->

考虑实值函数

$$f(x,y)=\begin{cases}
\dfrac{xy(x^2-y^2)}{x^2+y^2},&(x,y)\ne(0,0),\\[4pt]
0,&(x,y)=(0,0).
\end{cases}$$

<!-- bilingual-en:start -->
The function is zero on both coordinate axes. Fix $y\ne0$ and form the first difference quotient in $x$; fix $x\ne0$ and do the other one:
<!-- bilingual-en:end -->

函数在两条坐标轴上都为零。固定 $y\ne0$，先对 $x$ 作差商；再固定 $x\ne0$，作另一方向的差商：

$$f_x(0,y)=\lim_{h\to0}y\frac{h^2-y^2}{h^2+y^2}=-y,
\qquad
f_y(x,0)=\lim_{k\to0}x\frac{x^2-k^2}{x^2+k^2}=x.$$

<!-- bilingual-en:start -->
At the origin both first partials are zero, directly from the zero values on the axes. Therefore these two displayed axis formulas also hold when the fixed coordinate is zero. Now differentiate them once more:
<!-- bilingual-en:end -->

原点的一阶偏导都为零，这直接来自坐标轴上的函数值为零。因此上面两条轴上公式在固定坐标为零时也成立。再分别求一次导数，得到

$$f_{xy}(0,0)=\lim_{k\to0}\frac{-k-0}{k}=-1,
\qquad
f_{yx}(0,0)=\lim_{h\to0}\frac{h-0}{h}=1.$$

<!-- bilingual-en:start -->
Along the $x$-axis, $f_x(x,0)=0$; along the $y$-axis, $f_y(0,y)=0$. Hence the two pure second partials at the origin are zero as well. All four Hessian entries exist, but
<!-- bilingual-en:end -->

沿 $x$ 轴有 $f_x(x,0)=0$，沿 $y$ 轴有 $f_y(0,y)=0$，因此原点的两个纯二阶偏导也为零。四个 Hessian 元素全部存在，却有

$$H_f(0,0)=\begin{pmatrix}0&-1\\1&0\end{pmatrix}\ne H_f(0,0)^T.$$

<!-- bilingual-en:start -->
In particular, the continuity hypothesis of the theorem fails. For instance, at $(x,0)$ with $x\ne0$, the function is smooth and $f_{yx}(x,0)=1$, so $f_{xy}(x,0)=1$. This tends to $1$ along the punctured $x$-axis, whereas $f_{xy}(0,0)=-1$. Existing mixed partials cannot by themselves justify swapping their order.
<!-- bilingual-en:end -->

这里确实不满足定理的连续性前提。例如在 $x\ne0$ 的 $(x,0)$ 附近，函数光滑，且 $f_{yx}(x,0)=1$，所以 $f_{xy}(x,0)=1$。沿去掉原点的 $x$ 轴趋近时，这个值趋于 $1$，但 $f_{xy}(0,0)=-1$。混合偏导都存在，本身不足以交换求导次序。

<!-- bilingual-en:start -->
## Symmetrizing a quadratic form does not exchange derivatives
<!-- bilingual-en:end -->

## 它和二次型对称化不是同一件事

<!-- bilingual-en:start -->
For any real square matrix $A$ and real column vector $h$ of the matching size, [[二次型对称化|quadratic-form symmetrization]] gives
<!-- bilingual-en:end -->

对任意实方阵 $A$ 和尺寸匹配的实列向量 $h$，[[二次型对称化]]给出

$$h^TAh=h^T\frac{A+A^T}{2}h.$$

<!-- bilingual-en:start -->
The antisymmetric part contributes zero to this scalar expression. That algebra does not prove $A=A^T$. In the counterexample the symmetric part of $H_f(0,0)$ is zero, yet its mixed entries remain $-1$ and $1$. Replacing the matrix by its symmetric part does not change what the original derivatives were, and cannot supply missing regularity for a [[多元Taylor近似|second-order Taylor approximation]].
<!-- bilingual-en:end -->

反对称部分对这个标量表达式的贡献为零，但这个代数事实并不能证明 $A=A^T$。反例中 $H_f(0,0)$ 的对称部分为零，原有混合元素却仍是 $-1$ 和 $1$。把矩阵换成对称部分，不会改变原来求得的导数，也不能替[[多元Taylor近似|二阶 Taylor 近似]]补上缺失的正则条件。

> [!question]- What information allows the swap? · 什么信息允许交换次序？
> <!-- bilingual-en:start -->
> Knowing that $f_{xy}(a)$ and $f_{yx}(a)$ both exist is insufficient. The $C^2$ neighborhood condition above guarantees equality. If that condition fails, check the derivatives or another applicable theorem; failure of this sufficient condition alone does not prove inequality.
> <!-- bilingual-en:end -->
>
> 只知道 $f_{xy}(a)$ 与 $f_{yx}(a)$ 都存在还不够。上面的邻域 $C^2$ 条件可以保证相等；若它不成立，应直接检查导数或使用另一条适用的定理，不能仅凭这条充分条件失效就断言两者不等。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Lebl, Basic Analysis II, v6.3, §8.6, definition 8.6.1, proposition 8.6.2 and exercise 8.6.2, pp.60–62](https://www.jirka.org/ra/realanal2.pdf#page=60): $C^2$, the mixed-partial theorem, its two-stage mean-value proof, and the counterexample used here. The axis limits and full matrix are worked out above.
- [[01_Math/08_EC400_MathsCamp/01_Revision_Mathematics/Notes/By_Topic/Revision Maths Notes 7, Working with Multivariate Calculus.pdf#page=11|LSE EC400 Notes 7, §7.7, PDF p.11 / print p.69]]: the course's Young theorem name, derivative order, and continuous-second-partial sufficient condition. The quadratic identity is separately proved in the linked symmetrization card.
<!-- bilingual-en:end -->

- Lebl《Basic Analysis II》v6.3 §8.6，定义8.6.1、命题8.6.2与习题8.6.2，第60–62页：$C^2$、混合偏导定理、两次中值定理证明及本文反例；轴上极限和完整矩阵已在正文展开。
- LSE EC400 Notes 7 §7.7，PDF 第11页／印本69页：课程中的 Young 定理名称、求导顺序以及二阶偏导连续这一充分条件。二次型恒等式在所链接的对称化卡中另有证明。

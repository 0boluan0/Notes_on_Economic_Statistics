---
aliases:
  - Jacobian matrix
  - 雅可比矩阵
  - 雅克比矩阵
student_os: knowledge-atom
atom_id: CALC-MV-003
atom_set: multivariable-differentiation
atom_type: definition
status: source-checked
requires:
  - "[[全微分]]"
  - "[[线性映射与矩阵表示]]"
related:
  - "[[Jacobian行列式]]"
leads_to:
  - "[[多元链式法则]]"
  - "[[Hessian矩阵]]"
  - "[[隐函数定理]]"
  - "[[矩形Jacobian体积因子]]"
part_of:
  - "[[多元微分.canvas|多元微分]]"
  - "[[行列式.canvas|行列式]]"
---

<!-- bilingual-en:start -->
# Jacobian matrices
<!-- bilingual-en:end -->

*Jacobian 矩阵*

<!-- bilingual-en:start -->
A **Jacobian matrix** organizes how several outputs respond to several inputs. Each row follows one output, and each column follows one input. For a differentiable map, multiplying this matrix by a small input change predicts the corresponding output change. It is the coordinate representation of the [[全微分|total differential]], not a single number measuring volume.
<!-- bilingual-en:end -->

**Jacobian 矩阵**把多个输出对多个输入的变化率组织起来：每行对应一个输出，每列对应一个输入。映射可微时，用这个矩阵乘小的输入增量，就得到输出变化的一阶预测。它是[[全微分]]的坐标表示，不是一个单独测量体积的数。

<!-- bilingual-en:start -->
Let $F=(F_1,\ldots,F_m)^T:U\subseteq\mathbb R^n\to\mathbb R^m$, with $U$ open and $a\in U$. The positive integers $n,m$ count inputs and outputs; $T$ means [[矩阵转置|transpose]], so the components are arranged in columns. A [[偏导数|partial derivative]] $\partial F_i/\partial x_j$ changes input $j$ while holding the other inputs fixed. When all these partials exist at $a$, their array is
<!-- bilingual-en:end -->

设 $F=(F_1,\ldots,F_m)^T:U\subseteq\mathbb R^n\to\mathbb R^m$，$U$ 为开集，$a\in U$。正整数 $n,m$ 分别是输入数和输出数；$T$ 表示[[矩阵转置|转置]]，把分量排成列。[[偏导数]] $\partial F_i/\partial x_j$ 表示保持其余输入不变、改变第 $j$ 个输入时，第 $i$ 个输出的变化率。若这些偏导在 $a$ 都存在，它们组成

$$J_F(a)=\left[\frac{\partial F_i}{\partial x_j}(a)\right]_{\substack{1\le i\le m\\1\le j\le n}}
=\begin{pmatrix}
\partial_1F_1(a)&\cdots&\partial_nF_1(a)\\
\vdots&\ddots&\vdots\\
\partial_1F_m(a)&\cdots&\partial_nF_m(a)
\end{pmatrix}\in\mathbb R^{m\times n}.$$

<!-- bilingual-en:start -->
Thus a map with three inputs and two outputs has a $2\times3$ Jacobian. Its product with a $3\times1$ input increment is a $2\times1$ output prediction. The matrix records rates at a specified base point; changing that point usually changes its entries.
<!-- bilingual-en:end -->

因此，三个输入、两个输出的映射具有 $2\times3$ Jacobian；它乘 $3\times1$ 输入增量，得到 $2\times1$ 输出预测。矩阵记录的是指定基点处的变化率；改变基点，元素通常也会改变。

<!-- bilingual-en:start -->
## From partial derivatives to a local input-output map
<!-- bilingual-en:end -->

## Jacobian 做的是局部输入—输出映射

<!-- bilingual-en:start -->
The existence of the array alone does not establish a local approximation. If $F$ is differentiable at $a$, then its derivative $DF(a)$ is a linear map and $J_F(a)$ is its [[线性映射与矩阵表示|matrix in the standard input and output bases]]. For an input increment $h$,
<!-- bilingual-en:end -->

这张偏导表存在，还不能证明局部近似成立。若 $F$ 在 $a$ 可微，导数 $DF(a)$ 是线性映射，$J_F(a)$ 才是它在[[线性映射与矩阵表示|标准输入基与输出基下的矩阵]]。对输入增量 $h$，有

$$DF(a)h=J_F(a)h,\qquad
F(a+h)=F(a)+J_F(a)h+r(h),\qquad
\frac{\|r(h)\|_2}{\|h\|_2}\longrightarrow0.$$

<!-- bilingual-en:start -->
Why does column $j$ have these entries? Moving by $te_j$ changes only input $j$. Differentiability makes $[F(a+te_j)-F(a)]/t$ tend to $DF(a)e_j$. Component $i$ of this limit is $\partial_jF_i(a)$. A matrix's $j$th column is precisely its action on $e_j$. Then [[矩阵乘法|matrix multiplication]] adds the column responses with weights $h_j$, giving $(J_F(a)h)_i=\sum_j\partial_jF_i(a)h_j$.
<!-- bilingual-en:end -->

为什么第 $j$ 列正好是这些元素？移动 $te_j$ 只改变第 $j$ 个输入。可微性使 $[F(a+te_j)-F(a)]/t$ 趋于 $DF(a)e_j$，其第 $i$ 个分量就是 $\partial_jF_i(a)$。而矩阵第 $j$ 列恰好是它作用于 $e_j$ 的结果。随后，[[矩阵乘法]]按权重 $h_j$ 把各列的响应相加，得到 $(J_F(a)h)_i=\sum_j\partial_jF_i(a)h_j$。

> [!warning] A table of partials can exist without a derivative · 偏导表存在不保证全微分存在
> <!-- bilingual-en:start -->
> The function that is zero on the two coordinate axes and one everywhere else has a zero partial-derivative row at the origin, but is not even continuous there. Thus its partial table does not predict all nearby changes. See [[偏导不可推可微|the precise counterexample and sufficient continuity conditions]].
> <!-- bilingual-en:end -->
>
> 在两条坐标轴上取零、其他点取一的函数，在原点的偏导行是零，却连连续都不成立。因此，这张偏导表不能预测附近所有变化，见[[偏导不可推可微|具体反例与连续偏导的充分条件]]。

<!-- bilingual-en:start -->
For a complete calculation, take
<!-- bilingual-en:end -->

作一个完整计算，取

$$F(x,y)=\begin{pmatrix}x^2y\\x+y\end{pmatrix},\qquad
J_F(x,y)=\begin{pmatrix}2xy&x^2\\1&1\end{pmatrix}.$$

<!-- bilingual-en:start -->
These polynomial components are differentiable. At $(1,2)$, the first output is four times as sensitive to a change in $x$ as to the same-sized change in $y$; the second output adds both changes. With $h=(0.01,-0.02)^T$,
<!-- bilingual-en:end -->

这些多项式分量可微。在 $(1,2)$ 处，第一个输出对 $x$ 增量的局部响应系数为 $4$、对 $y$ 的系数为 $1$；第二个输出把两个增量相加。取 $h=(0.01,-0.02)^T$，有

$$J_F(1,2)h=
\begin{pmatrix}4&1\\1&1\end{pmatrix}
\begin{pmatrix}0.01\\-0.02\end{pmatrix}
=\begin{pmatrix}0.02\\-0.01\end{pmatrix},\qquad
F(1.01,1.98)-F(1,2)=\begin{pmatrix}0.019798\\-0.01\end{pmatrix}.$$

<!-- bilingual-en:start -->
The first-component error is $-0.000202$; the second component is already linear, so its error is zero. For general small $h$, expanding gives $r(h)=(2h_1^2+2h_1h_2+h_1^2h_2,0)^T$. When $\|h\|_2\le1$, its length is at most $4\|h\|_2^2$, which verifies the required limit directly.
<!-- bilingual-en:end -->

第一分量的误差为 $-0.000202$；第二分量本来就是线性的，因此误差为零。对一般小增量展开可得 $r(h)=(2h_1^2+2h_1h_2+h_1^2h_2,0)^T$。当 $\|h\|_2\le1$ 时，其长度不超过 $4\|h\|_2^2$，直接验证了所需极限。

<!-- bilingual-en:start -->
## A scalar derivative is a row; the gradient is a column
<!-- bilingual-en:end -->

## 标量函数的行导数与列梯度

<!-- bilingual-en:start -->
When $m=1$, the matrix is one row. Under the column convention for the gradient,
<!-- bilingual-en:end -->

当 $m=1$ 时，这个矩阵只有一行。采用梯度为列的约定时，

$$J_f(a)=\begin{pmatrix}\partial_1f(a)&\cdots&\partial_nf(a)\end{pmatrix}
=\nabla f(a)^T,\qquad Df(a)h=J_f(a)h.$$

<!-- bilingual-en:start -->
Thus the gradient and the Jacobian contain the same coefficients in different shapes. The derivative $Df(a)$ is the underlying linear rule; identifying it with the row matrix presupposes these coordinates. This convention keeps the dimensions consistent when applying the [[多元链式法则|multivariable chain rule]].
<!-- bilingual-en:end -->

因此，梯度与 Jacobian 装着同一组系数，只是形状不同；导数 $Df(a)$ 是背后的线性规则，把它直接写成行矩阵，是在这些坐标约定之下的简写。这一约定能让[[多元链式法则]]中的维度保持一致。

<!-- bilingual-en:start -->
## A matrix and its determinant are different objects
<!-- bilingual-en:end -->

## 行列式不是每个 Jacobian 都有的属性

<!-- bilingual-en:start -->
Only a square Jacobian, with $m=n$, has a [[Jacobian行列式|Jacobian determinant]] $\det J_F(a)$, which is a scalar. For example, the matrix above at $(1,2)$ has determinant $4(1)-1(1)=3$, but the number $3$ cannot replace the whole matrix when predicting two output changes. A $2\times3$ Jacobian has no ordinary determinant at all.
<!-- bilingual-en:end -->

只有 $m=n$ 的方阵 Jacobian 才有[[Jacobian行列式]] $\det J_F(a)$，它是一个标量。比如上例在 $(1,2)$ 处的行列式为 $4(1)-1(1)=3$，但预测两个输出增量时，数 $3$ 不能代替整个矩阵。$2\times3$ Jacobian 则根本没有通常意义的行列式。

<!-- bilingual-en:start -->
For a differentiable parametrization with fewer inputs than outputs, the [[矩形Jacobian体积因子|square root of a Gram determinant]] describes local lower-dimensional volume under its stated rank conditions. Square Jacobians instead enter the [[逆函数定理|inverse function theorem]], [[隐函数定理|implicit function theorem]] and [[多元换元公式|change-of-variables theorem]] under those theorems' additional hypotheses. These applications do not redefine the Jacobian matrix.
<!-- bilingual-en:end -->

对输入维数少于输出维数的可微参数化，[[矩形Jacobian体积因子|Gram 行列式的平方根]]在相应秩条件下描述局部低维体积；方阵 Jacobian 则在各定理的额外前提下用于[[逆函数定理]]、[[隐函数定理]]和[[多元换元公式]]。这些应用并不改变 Jacobian 矩阵本身的定义。

<!-- visual-lab:links:start -->
## 交互理解

- [[07_Programme/可视化实验室/实验/jacobian|非线性变形怎样被局部Jacobian近似]]
<!-- visual-lab:links:end -->

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 18.S096, Derivatives as Linear Operators, §§2.3–2.4, PDF pp.4–7 / print 12–15](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec02.pdf#page=4): column gradients, output-by-input dimensions, partial entries, and linear-map action.
- [Lebl, Basic Analysis II, proposition 8.3.9, pp.39–40](https://www.jirka.org/ra/realanal2.pdf#page=39): differentiability implies the displayed coordinate representation, and existence of partials alone is insufficient.
- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/02_SOFP/Lectures/EC400 Lecture Notes SOFP.pdf#page=41|LSE EC400 SOFP Lecture Notes, p.41]]: equations as rows and solved variables as columns in an implicit system, with a separate nonzero determinant condition. The two-output example, exact error and dimensions above are independently calculated.
<!-- bilingual-en:end -->

- MIT 18.S096《Derivatives as Linear Operators》§§2.3–2.4（PDF 第4–7页，印本12–15页）：列梯度、输出行/输入列的尺寸、偏导元素与线性映射作用。
- Lebl《Basic Analysis II》命题8.3.9，第39–40页：可微推出所示坐标表示，但仅有偏导还不够。
- LSE EC400 SOFP 讲义第41页：隐式系统按方程排成行、待解变量排成列，并另外要求行列式非零。本文两输出算例、精确误差及尺寸独立计算。

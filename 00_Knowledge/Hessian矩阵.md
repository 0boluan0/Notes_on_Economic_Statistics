---
aliases:
  - "Hessian matrix"
student_os: knowledge-atom
atom_id: CALC-MV-008
atom_set: multivariable-differentiation
atom_type: definition
status: source-checked
requires:
  - "[[Jacobian矩阵]]"
leads_to:
  - "[[Hessian对称条件]]"
  - "[[Hessian方向二阶导数]]"
part_of:
  - "[[多元微分.canvas|多元微分]]"
---

<!-- bilingual-en:start -->
# Hessian matrices
<!-- bilingual-en:end -->

*Hessian 矩阵*

<!-- bilingual-en:start -->
A **Hessian matrix** records how a real-valued function's first-order rates change as its inputs move. A [[偏导数|partial derivative]] measures the rate when one input varies and the others stay fixed. Differentiating those rates again gives the second partial derivatives; the Hessian arranges all of them in one square table. It includes interactions between different inputs, not just the curvature along each coordinate axis.
<!-- bilingual-en:end -->

**Hessian 矩阵**记录实值函数的一阶变化率怎样随输入改变。[[偏导数]]测量只改变一个输入、固定其余输入时的变化率；再对这些变化率求偏导，得到二阶偏导。Hessian 把它们全部排成一张方形表格，既包含沿各坐标轴的二阶变化，也包含不同输入之间的交叉作用。

<!-- bilingual-en:start -->
For example, if $f(x,y)=x^2y$, the rate with respect to $x$ is $f_x=2xy$. Changing $x$ changes this rate at rate $2y$, while changing $y$ changes it at rate $2x$. Both quantities belong in the Hessian. We will compute the complete matrix below.
<!-- bilingual-en:end -->

例如，$f(x,y)=x^2y$ 关于 $x$ 的变化率为 $f_x=2xy$。改变 $x$ 时，这个变化率以 $2y$ 的速率改变；改变 $y$ 时，它以 $2x$ 的速率改变。两者都属于 Hessian，下面会算出整张矩阵。

<!-- bilingual-en:start -->
Let $f:U\to\mathbb R$, where $U$ is an [[开集|open subset]] of $\mathbb R^n$, $n\ge1$, and $a\in U$. Thus small changes of every coordinate are possible on both sides of $a$. Suppose the first partials $\partial_i f$ exist near $a$ and all their coordinate derivatives exist at $a$ as finite real numbers. Our convention is
<!-- bilingual-en:end -->

设 $f:U\to\mathbb R$，其中 $U$ 是 $\mathbb R^n$ 中的[[开集]]，$n\ge1$，$a\in U$。因此在 $a$ 附近，各个坐标都可以向正、负两侧作小幅变化。假设一阶偏导 $\partial_i f$ 在 $a$ 附近存在，且它们关于各坐标的偏导在 $a$ 都存在并为有限实数。本文采用

$$H_f(a)=\nabla^2f(a)=\bigl[f_{ij}(a)\bigr]_{i,j=1}^n,
\qquad f_{ij}=\partial_j(\partial_i f)
=\frac{\partial^2f}{\partial x_j\partial x_i}.$$

<!-- bilingual-en:start -->
Entry $(i,j)$ is in row $i$, column $j$: first differentiate $f$ with respect to input $i$, then differentiate that result with respect to input $j$. Thus the subscript $f_{xy}$ means $\partial_y(\partial_xf)$; in the nested operator or fraction, the rightmost differentiation acts first. The diagonal entries $f_{ii}$ repeat one input. The off-diagonal entries $f_{ij}$, $i\ne j$, are **mixed partials**. For one input the matrix has one entry, $f''(a)$.
<!-- bilingual-en:end -->

第 $(i,j)$ 个元素位于第 $i$ 行、第 $j$ 列：先对输入 $i$ 求导，再对所得结果关于输入 $j$ 求导。因此，下标 $f_{xy}$ 表示 $\partial_y(\partial_xf)$；嵌套算子或分式记号则由最右侧的求导先作用。对角元 $f_{ii}$ 对同一个输入求两次导；非对角元 $f_{ij}$（$i\ne j$）称为**混合偏导**。只有一个输入时，矩阵只有一个元素 $f''(a)$。

<!-- bilingual-en:start -->
## Why it is the Jacobian of the gradient
<!-- bilingual-en:end -->

## 为什么它是梯度的 Jacobian

<!-- bilingual-en:start -->
The [[全微分#实值函数、梯度与全微分|gradient]] is the column of first partial derivatives. Regard it as a new map from $n$ inputs to $n$ outputs:
<!-- bilingual-en:end -->

[[全微分#实值函数、梯度与全微分|梯度]]把一阶偏导排成一列。现在把它看作一个有 $n$ 个输入、$n$ 个输出的新映射：

$$\nabla f(x)=\begin{pmatrix}\partial_1f(x)\\\vdots\\\partial_nf(x)\end{pmatrix}.$$

<!-- bilingual-en:start -->
A [[Jacobian矩阵|Jacobian matrix]] puts the derivative of output $i$ with respect to input $j$ in position $(i,j)$. For this map that entry is $\partial_j(\partial_i f)$, so its partial-derivative array is exactly $H_f(a)$. If the gradient map is also [[全微分|differentiable at $a$]], the matrix represents its total derivative: the same linear rule approximates its change for every small input vector $h$.
<!-- bilingual-en:end -->

[[Jacobian矩阵]]的第 $(i,j)$ 项，是输出 $i$ 关于输入 $j$ 的偏导。对梯度映射，这一项就是 $\partial_j(\partial_i f)$，所以它的偏导数组恰好为 $H_f(a)$。如果梯度映射还[[全微分|在 $a$ 可微]]，该矩阵就表示它的全微分：同一个线性规则可以近似每个小输入向量 $h$ 所造成的梯度变化。

$$J_{\nabla f}(a)=H_f(a),\qquad
\nabla f(a+h)=\nabla f(a)+H_f(a)h+r(h),\qquad
\frac{\|r(h)\|_2}{\|h\|_2}\longrightarrow0\quad(h\to0,\ h\ne0).$$

<!-- bilingual-en:start -->
Here $r(h)$ is the remaining vector error, and $\|\cdot\|_2$ is [[欧氏范数|Euclidean length]]. The product $H_f(a)h$ predicts the change of the **gradient**, not the change of $f$ itself. Merely having all matrix entries does not establish this approximation. Likewise, those entries need not be symmetric merely because they are second partials: [[Hessian对称条件|continuity assumptions give a standard sufficient condition for symmetry]].
<!-- bilingual-en:end -->

$r(h)$ 是剩余的向量误差，$\|\cdot\|_2$ 表示[[欧氏范数|欧氏长度]]。乘积 $H_f(a)h$ 预测的是**梯度**的变化，而不是 $f$ 本身的变化。仅有全部矩阵元素，还不能证明这个近似成立；仅仅因为它们都是二阶偏导，也不能断言矩阵对称。[[Hessian对称条件|连续性假设给出一条常用的对称性充分条件]]。

<!-- bilingual-en:start -->
## Mixed partials describe how one rate responds to another input
<!-- bilingual-en:end -->

## 交叉偏导记录坐标之间的局部作用

<!-- bilingual-en:start -->
For the opening polynomial, first compute both first partials, then differentiate each of them with respect to both inputs:
<!-- bilingual-en:end -->

对开头的多项式，先算两个一阶偏导，再分别对两个输入求导：

$$f(x,y)=x^2y,\qquad
\nabla f(x,y)=\begin{pmatrix}2xy\\x^2\end{pmatrix},\qquad
H_f(x,y)=\begin{pmatrix}2y&2x\\2x&0\end{pmatrix}.$$

<!-- bilingual-en:start -->
The upper-right entry $2x$ tells us how the $x$-rate changes with $y$. The lower-left entry tells us how the $y$-rate changes with $x$. They agree for this smooth polynomial, but arise from different differentiation orders. At the point $(1,2)$, the matrix is $\begin{pmatrix}4&2\\2&0\end{pmatrix}$: changing $y$ has zero effect on the $y$-rate $x^2$, yet it does change the $x$-rate $2xy$.
<!-- bilingual-en:end -->

右上角的 $2x$ 说明 $x$ 方向变化率怎样随 $y$ 改变；左下角说明 $y$ 方向变化率怎样随 $x$ 改变。对于这个光滑多项式，两者相等，但求导顺序不同。在 $(1,2)$ 处，矩阵为 $\begin{pmatrix}4&2\\2&0\end{pmatrix}$：改变 $y$ 不影响 $y$ 方向变化率 $x^2$，却会影响 $x$ 方向变化率 $2xy$。

> [!example] Predicting a change of the gradient · 预测梯度的变化
> <!-- bilingual-en:start -->
> Take $h=(0.01,-0.02)^T$, where $T$ denotes [[矩阵转置|transpose]]. The matrix prediction and the actual change are
> <!-- bilingual-en:end -->
>
> 取 $h=(0.01,-0.02)^T$，上标 $T$ 表示[[矩阵转置|转置]]。矩阵预测与真实变化分别为
>
> $$H_f(1,2)h=\begin{pmatrix}0\\0.02\end{pmatrix},\qquad
> \nabla f(1.01,1.98)-\nabla f(1,2)=\begin{pmatrix}-0.0004\\0.0201\end{pmatrix}.$$
>
> <!-- bilingual-en:start -->
> For a general $h=(h_1,h_2)^T$, the exact remaining error is $r(h)=(2h_1h_2,h_1^2)^T$. Since $\|r(h)\|_2\le2\|h\|_2^2$, its ratio to $\|h\|_2$ tends to zero. The prediction is a first-order approximation, not an exact finite-step identity.
> <!-- bilingual-en:end -->
>
> 对一般 $h=(h_1,h_2)^T$，精确余项为 $r(h)=(2h_1h_2,h_1^2)^T$。因为 $\|r(h)\|_2\le2\|h\|_2^2$，它与 $\|h\|_2$ 的比趋于零。这个预测是一阶近似，不是有限步长下的精确恒等式。

<!-- bilingual-en:start -->
Combining Hessian entries into a scalar second derivative along a line has its own conditions and proof in [[Hessian方向二阶导数|the Hessian formula for second directional derivatives]]. That use of the matrix should not be inferred from the existence of an arbitrary table of second partials alone.
<!-- bilingual-en:end -->

把 Hessian 各元素组合成沿直线的标量二阶导数，另有条件与证明，见[[Hessian方向二阶导数]]。不能只凭一张二阶偏导表存在，就自动断言这种用法成立。

<!-- bilingual-en:start -->
## Several outputs require several component Hessians
<!-- bilingual-en:end -->

## 向量值函数的二阶导数不再是一张 Hessian

<!-- bilingual-en:start -->
The Hessian just defined belongs to a real-valued function. For $F=(F_1,\ldots,F_m):U\subseteq\mathbb R^n\to\mathbb R^m$ with $m>1$, each scalar component $F_k$ can have its own $n\times n$ Hessian. For example, $F(x,y)=(x^2y,x+y)$ has the matrix above for its first component and the zero $2\times2$ matrix for its second component.
<!-- bilingual-en:end -->

这里定义的 Hessian 属于实值函数。对 $F=(F_1,\ldots,F_m):U\subseteq\mathbb R^n\to\mathbb R^m$，若 $m>1$，每个标量分量 $F_k$ 都可以有自己的 $n\times n$ Hessian。例如 $F(x,y)=(x^2y,x+y)$ 的第一分量有上面的矩阵，第二分量有零 $2\times2$ 矩阵。

<!-- bilingual-en:start -->
If the full second derivative exists, it can be viewed as a **bilinear map**: it takes two input vectors, returns an $m$-component vector, and is linear in either input when the other is fixed. Its coordinate data have three indices: the output component and the two differentiated inputs. Thus a scalar function on $\mathbb R^3$ has a $3\times3$ Hessian with entry $(2,3)$ equal to $\partial_3(\partial_2f)$; a map from $\mathbb R^3$ to $\mathbb R^2$ needs two such matrices. A single $n\times n$ Hessian cannot hold all $m$ components: keep $m$ component Hessians, or reshape the same data with a stated indexing convention.
<!-- bilingual-en:end -->

如果完整的二阶导数存在，可以把它看成一个**双线性映射**：输入两个向量，输出一个 $m$ 分量向量，固定其中一个输入时对另一个输入线性。其坐标数据有三个指标：输出分量以及两次求导的输入。因此，$\mathbb R^3$ 上的实值函数具有 $3\times3$ Hessian，第 $(2,3)$ 项为 $\partial_3(\partial_2f)$；从 $\mathbb R^3$ 到 $\mathbb R^2$ 的映射则需要两张这样的矩阵。单独一张 $n\times n$ Hessian 装不下全部 $m$ 个分量；可以保留 $m$ 张分量 Hessian，也可以在说明指标排列约定后，把同一批数据重新排成其他尺寸的矩阵。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/08_EC400_MathsCamp/01_Revision_Mathematics/Notes/By_Topic/Revision Maths Notes 7, Working with Multivariate Calculus.pdf#page=11|LSE EC400 Notes 7, §7.7, PDF p.11 / print p.69]]: course notation, pure and mixed second partials, the Hessian array and the continuity qualification for symmetry.
- [MIT 18.S096, Second Derivatives, Bilinear Maps, and Hessian Matrices, §§12.1–12.2, PDF pp.1–4 / print pp.87–90](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec12.pdf#page=1): gradient Jacobian, local change of the gradient, and vector-output second derivatives. Here finite changes retain an explicit remainder.
- [Lebl, Basic Analysis II, v6.3, §8.6, pp.60–62](https://www.jirka.org/ra/realanal2.pdf#page=60): derivative-order notation and the continuity assumptions for symmetry. The polynomial matrix, dimensions and finite-step errors above are calculated directly.
<!-- bilingual-en:end -->

- LSE EC400 Notes 7 §7.7，PDF 第11页／印本69页：课程记号、纯与混合二阶偏导、Hessian 数组及对称性的连续性限定。
- MIT 18.S096《Second Derivatives, Bilinear Maps, and Hessian Matrices》§§12.1–12.2，PDF 第1–4页／印本87–90页：梯度的 Jacobian、梯度局部变化以及向量输出的二阶导数；本文对有限增量明确保留余项。
- Lebl《Basic Analysis II》v6.3 §8.6，第60–62页：求导次序记号和对称性的连续性条件。本文多项式矩阵、尺寸与有限步误差直接计算。

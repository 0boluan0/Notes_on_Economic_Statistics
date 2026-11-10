---
aliases:
  - Jacobian determinant
  - 雅可比行列式
student_os: knowledge-atom
atom_id: LA-DET-015
atom_set: determinants
atom_type: definition
status: source-checked
requires:
  - "[[行列式体积与取向]]"
  - "[[Jacobian矩阵]]"
leads_to:
  - "[[多元换元公式]]"
  - "[[逆函数定理]]"
related:
  - "[[Jacobian奇异不推局部非单射]]"
  - "[[矩形Jacobian体积因子]]"
part_of:
  - "[[行列式.canvas]]"
  - "[[多元微分.canvas|多元微分]]"
---

<!-- bilingual-en:start -->
# Jacobian determinants
<!-- bilingual-en:end -->

*Jacobian 行列式*

<!-- bilingual-en:start -->
A **Jacobian determinant** is one number extracted from the derivative matrix of a map with equally many inputs and outputs. It describes how the local linear approximation scales signed volume. Its absolute value gives the ordinary, nonnegative volume factor; its sign records whether the ordered coordinate directions keep or reverse orientation.
<!-- bilingual-en:end -->

**Jacobian 行列式**是从输入、输出维数相同的映射的导数矩阵中提取的一个数。它描述局部线性近似怎样缩放有向体积：绝对值给出非负的普通体积因子，符号记录有序坐标方向的取向保持还是翻转。

<!-- bilingual-en:start -->
For a concrete example, let $F(x,y)=(2x+y,-3y)$. Its [[Jacobian矩阵|Jacobian matrix]] has one row per output and one column per input. Its [[行列式|determinant]] is
<!-- bilingual-en:end -->

例如，取 $F(x,y)=(2x+y,-3y)$。它的 [[Jacobian矩阵]]按输出排成行、按输入排成列，[[行列式]]为

$$J_F=\begin{pmatrix}2&1\\0&-3\end{pmatrix},\qquad
\det J_F=2(-3)-1(0)=-6.$$

<!-- bilingual-en:start -->
The unit square becomes the parallelogram with edges $(2,0)$ and $(1,-3)$: base length $2$, perpendicular height $3$, hence area $6$. The negative sign says that orientation reverses; the area is not negative. Because this map is linear, the factor is exact at every scale. For a nonlinear map, the derivative generally supplies only a first-order description.
<!-- bilingual-en:end -->

单位正方形变成由 $(2,0)$ 与 $(1,-3)$ 两条边围成的平行四边形：底长为 $2$，垂直高度为 $3$，面积为 $6$。负号表示取向翻转，并不表示面积为负。这个映射是线性的，所以该因子在任何尺度都精确成立；非线性映射的导数通常只提供一阶描述。

<!-- bilingual-en:start -->
## Define the number at a specified point
<!-- bilingual-en:end -->

## 在指定点定义这个数

<!-- bilingual-en:start -->
Let $n\ge1$, let $U\subseteq\mathbb R^n$ be open, and let $F=(F_1,\ldots,F_n):U\to\mathbb R^n$. Fix a point $a\in U$ and the order of the input and output coordinates. If all first [[偏导数|partial derivatives]] exist at $a$, the square array $J_F(a)=[\frac{\partial F_i}{\partial x_j}(a)]$ has a determinant:
<!-- bilingual-en:end -->

设 $n\ge1$，$U\subseteq\mathbb R^n$ 为开集，$F=(F_1,\ldots,F_n):U\to\mathbb R^n$。固定点 $a\in U$，以及输入、输出坐标的顺序。若 $a$ 处所有一阶[[偏导数]]都存在，方阵 $J_F(a)=[\frac{\partial F_i}{\partial x_j}(a)]$ 就有行列式：

$$\boxed{\operatorname{Jac}F(a)=\det J_F(a).}$$

<!-- bilingual-en:start -->
This scalar is the Jacobian determinant. The geometry additionally requires [[全微分|differentiability]] at $a$: a single linear map must approximate changes in every direction. In standard coordinates that map is $J_F(a)$, and
<!-- bilingual-en:end -->

这个标量就是 Jacobian 行列式。几何解释还需要 $F$ 在 $a$ 处[[全微分|可微]]：同一个线性映射必须近似所有方向上的变化。在标准坐标下，这个映射由 $J_F(a)$ 表示，并满足

$$F(a+h)=F(a)+J_F(a)h+r(h),\qquad
\frac{\|r(h)\|_2}{\|h\|_2}\longrightarrow0\quad(h\to0,\ h\ne0).$$

<!-- bilingual-en:start -->
Here $h$ is a small input displacement, $r(h)$ is the approximation error, and $\|\cdot\|_2$ is [[欧氏范数|Euclidean length]]. The [[行列式体积与取向|linear determinant-volume rule]] applied to $h\mapsto J_F(a)h$ now gives the signed factor $\det J_F(a)$ and ordinary factor $|\det J_F(a)|$. Translation by $F(a)$ changes position but not volume. Existence of the partial-derivative array alone does not justify this linear approximation.
<!-- bilingual-en:end -->

这里 $h$ 是小的输入位移，$r(h)$ 是近似误差，$\|\cdot\|_2$ 是[[欧氏范数|欧氏长度]]。对线性映射 $h\mapsto J_F(a)h$ 使用[[行列式体积与取向|行列式的体积规则]]，便得到有向因子 $\det J_F(a)$ 与普通因子 $|\det J_F(a)|$。平移 $F(a)$ 改变位置，但不改变体积。仅有偏导数组成的表，并不能保证这项线性近似。

<!-- bilingual-en:start -->
## Local scaling is not an exact finite-region formula
<!-- bilingual-en:end -->

## 局部缩放不是有限区域的精确公式

<!-- bilingual-en:start -->
Take $F(x,y)=(x^2,y)$ near $a=(1,0)$. Its Jacobian is $\begin{pmatrix}2x&0\\0&1\end{pmatrix}$, so the factor at $a$ is $2$. For positive $h,k$, the rectangle $[1,1+h]\times[0,k]$ maps one-to-one onto $[1,(1+h)^2]\times[0,k]$. The exact areas give
<!-- bilingual-en:end -->

在 $a=(1,0)$ 附近取 $F(x,y)=(x^2,y)$。Jacobian 为 $\begin{pmatrix}2x&0\\0&1\end{pmatrix}$，因此 $a$ 处的因子为 $2$。对正数 $h,k$，矩形 $[1,1+h]\times[0,k]$ 一一映到 $[1,(1+h)^2]\times[0,k]$，精确面积之比为

$$\frac{\text{output area}}{\text{input area}}
=\frac{(2h+h^2)k}{hk}=2+h.$$

<!-- bilingual-en:start -->
As the rectangle shrinks, this approaches $2$; at $h=0.1$ it is $2.1$. For an entire region, the [[多元换元公式|change-of-variables theorem]] integrates the varying absolute determinant and checks its own hypotheses, such as a $C^1$ one-to-one correspondence with a $C^1$ inverse on the stated open domains, and integrability. Here $C^1$ means continuously differentiable. A determinant calculated at one point does not establish those global conditions.
<!-- bilingual-en:end -->

矩形缩小时，这个比值趋于 $2$；而 $h=0.1$ 时它为 $2.1$。对整个区域，[[多元换元公式]]需要积分随位置变化的绝对行列式，并核对自己的条件，例如所述开区域间为 $C^1$ 一一对应、逆映射也为 $C^1$，以及被积函数可积。这里 $C^1$ 指连续可微。在一点算出行列式，并不能证明这些整体条件。

> [!warning] Zero first-order volume does not settle the nonlinear map · 一阶体积为零不能判定非线性映射
> <!-- bilingual-en:start -->
> If $\det J_F(a)=0$, the derivative has [[矩阵秩|rank]] below $n$: its columns span fewer than $n$ independent directions. Its full-dimensional linear volume is zero. Yet $F(t)=t^3$ has derivative zero at $0$ and is still one-to-one. The separate [[Jacobian奇异不推局部非单射|singular-Jacobian boundary]] explains why this is not a converse to the [[逆函数定理|inverse function theorem]].
> <!-- bilingual-en:end -->
>
> 若 $\det J_F(a)=0$，导数的[[矩阵秩|秩]]小于 $n$：各列张成的独立方向不足 $n$ 个，其满维线性体积为零。但 $F(t)=t^3$ 在 $0$ 处导数为零，仍然一一。[[Jacobian奇异不推局部非单射]]说明了为什么不能据此反用[[逆函数定理]]。

<!-- bilingual-en:start -->
## Dimensions and coordinate order
<!-- bilingual-en:end -->

## 尺寸与顺序边界

<!-- bilingual-en:start -->
A map with $n$ inputs and $m$ outputs has an $m\times n$ Jacobian. If $m\ne n$, this matrix has no ordinary determinant. When $m\ge n$, the [[矩形Jacobian体积因子|Gram volume factor]] measures the derivative’s $n$-dimensional volume instead. For a square Jacobian, swapping two input coordinates swaps two columns; swapping two output coordinates swaps two rows. Either single swap changes the determinant’s sign and preserves its absolute value. For the opening matrix, swapping its columns changes $-6$ into $6$, with the same area factor.
<!-- bilingual-en:end -->

有 $n$ 个输入、$m$ 个输出的映射具有 $m\times n$ Jacobian。若 $m\ne n$，它没有通常意义的行列式。$m\ge n$ 时，[[矩形Jacobian体积因子]]可测量导数的 $n$ 维体积。方阵情形中，交换两个输入坐标就是交换两列，交换两个输出坐标就是交换两行；单独做其中一次，行列式变号，绝对值不变。开头矩阵交换两列后，$-6$ 变成 $6$，面积因子仍相同。

<!-- bilingual-en:start -->
These sign statements use real coordinates with chosen orientations. A complex determinant has no positive/negative orientation sign; a complex-linear map interpreted in real coordinates has a different real dimension. Keep the coordinate field and the volume being measured explicit.
<!-- bilingual-en:end -->

以上正负号的解释使用已选取向的实坐标。复行列式没有“正或负”的取向符号；把复线性映射视为实映射时，实维数也会改变。必须明确所用数域以及测量哪一种体积。

<!-- visual-lab:links:start -->
## 交互理解

- [[07_Programme/可视化实验室/实验/jacobian|非线性变形怎样被局部Jacobian近似]]
<!-- visual-lab:links:end -->

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Lebl, *Basic Analysis II*, §8.3, definition 8.3.1 and proposition 8.3.9, PDF pp.35 and 39–40](https://www.jirka.org/ra/realanal2.pdf#page=35): derivative as a uniform linear approximation and its partial-derivative matrix.
- [Eliashberg, Stanford Math 52H, §§3.2–3.3, pp.31–33](https://math.stanford.edu/~eliash/Public/52h-2013/52h-2013/52htext.pdf#page=31), together with [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.7sum.pdf#page=2|MIT 18.06SC Session 2.7, p.2]]: absolute determinant as parallelepiped volume; orientation is kept distinct from ordinary volume.
- [MIT 18.101, *Integrating Densities on Manifolds*, p.1, theorem 1](https://math.mit.edu/classes/18.101/fa07/pub/int-density.pdf#page=1): the additional diffeomorphism conditions in global change of variables. The linear example, finite-rectangle ratio, coordinate swap and cubic boundary above are checked directly.
<!-- bilingual-en:end -->

- Lebl《Basic Analysis II》§8.3，定义8.3.1及命题8.3.9，PDF第35、39–40页：核对统一线性近似与偏导矩阵表示。
- Stanford Math52H §§3.2–3.3，第31–33页，以及MIT18.06SC第2.7节摘要第2页：核对绝对行列式与平行多面体体积，并区分取向和普通体积。
- MIT18.101《Integrating Densities on Manifolds》第1页定理1：核对整体换元的微分同胚条件。线性例、有限矩形比值、坐标交换及立方函数边界均直接计算。

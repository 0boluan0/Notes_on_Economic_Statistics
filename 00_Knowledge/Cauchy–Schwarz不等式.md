---
student_os: knowledge-atom
atom_id: OPT-TOOLS-036
aliases:
- Cauchy–Schwarz inequality
- Cauchy-Schwarz inequality
- 柯西施瓦茨不等式
status: source-checked
---

<!-- bilingual-en:start -->
# Cauchy–Schwarz inequality
<!-- bilingual-en:end -->

Cauchy–Schwarz不等式（实向量）

<!-- bilingual-en:start -->
The Cauchy–Schwarz inequality bounds the dot product of two vectors using only their lengths. A dot product adds products of matching coordinates; it becomes large in magnitude when the vectors point along the same line. The inequality says that no arrangement of directions can make its magnitude exceed the product of the lengths.
<!-- bilingual-en:end -->

Cauchy–Schwarz不等式只利用两个向量的长度，就能控制它们的点积。点积把对应坐标相乘再相加；两个向量沿同一条直线时，点积的绝对大小可以达到上限。这个不等式说明：无论方向如何安排，都不能超过长度的乘积。

<!-- bilingual-en:start -->
For $u=(u_1,\ldots,u_n)$ and $v=(v_1,\ldots,v_n)$ in $\mathbb R^n$, their [[向量点积|dot product]] is $u^Tv=\sum_{i=1}^nu_iv_i$. The transpose makes the column vector into a row, as in [[矩阵转置|matrix transposition]]. Their [[欧氏范数|Euclidean lengths]] are $\|u\|_2=\sqrt{\sum_i u_i^2}$ and $\|v\|_2=\sqrt{\sum_i v_i^2}$. The subscript 2 records the use of squares; the outer square root returns to the original length scale.
<!-- bilingual-en:end -->

设$u=(u_1,\ldots,u_n)$、$v=(v_1,\ldots,v_n)$属于$\mathbb R^n$，[[向量点积|点积]]为$u^Tv=\sum_{i=1}^nu_iv_i$。这里的转置将列向量变成行向量，见[[矩阵转置]]。[[欧氏范数|欧氏长度]]分别为$\|u\|_2=\sqrt{\sum_i u_i^2}$和$\|v\|_2=\sqrt{\sum_i v_i^2}$；下标2表示使用平方，外面的平方根把结果还原到长度尺度。

> [!important] The bound and its equality cases · 不等式与等号条件
> <!-- bilingual-en:start -->
> For any real vectors of the same dimension, $|u^Tv|\le\|u\|_2\|v\|_2$. Equality holds exactly when the two vectors are linearly dependent (the opposite of [[线性无关|linear independence]]): one is a scalar multiple of the other, including the cases where either vector is zero. For nonzero vectors, both the same direction and the opposite direction attain equality because the left side is an absolute value.
> <!-- bilingual-en:end -->
>
> 对任意同维实向量，有$|u^Tv|\le\|u\|_2\|v\|_2$。等号成立当且仅当两个向量线性相关（不满足[[线性无关]]），即其中一个是另一个的标量倍，包括任一向量为零的情况。对非零向量，同向和反向都能取等号，因为左边有绝对值。

<!-- bilingual-en:start -->
## Two vectors with the same lengths can give different products
<!-- bilingual-en:end -->

长度相同，点积仍可不同

<!-- bilingual-en:start -->
Take $u=(1,2)$ and $v=(2,-1)$. Both have length $\sqrt5$, while $u^Tv=2-2=0$. They are [[正交|orthogonal]], and the bound reads $0\le5$. Instead choose $w=3u=(3,6)$. Then $u^Tw=15$ and $\|u\|_2\|w\|_2=\sqrt5\sqrt{45}=15$, so equality holds. With $w=-3u$, the dot product is $-15$, but its absolute value still attains the bound.
<!-- bilingual-en:end -->

取$u=(1,2)$、$v=(2,-1)$，二者长度都是$\sqrt5$，但$u^Tv=2-2=0$。它们[[正交]]，不等式为$0\le5$。若改取$w=3u=(3,6)$，则$u^Tw=15$，而$\|u\|_2\|w\|_2=\sqrt5\sqrt{45}=15$，达到等号。取$w=-3u$时点积为$-15$，绝对值仍达到上界。

<!-- bilingual-en:start -->
## Subtract the best scalar multiple
<!-- bilingual-en:end -->

减去最合适的标量倍

<!-- bilingual-en:start -->
If $v=0$, both sides of the inequality are zero. Otherwise choose $\lambda=(u^Tv)/\|v\|_2^2$. The squared length of $u-\lambda v$ cannot be negative. Expanding its coordinates gives
<!-- bilingual-en:end -->

$v=0$时两边均为零。否则取$\lambda=(u^Tv)/\|v\|_2^2$。$u-\lambda v$的长度平方不能为负，展开得到

$$
0\le\|u-\lambda v\|_2^2
=\|u\|_2^2-2\lambda u^Tv+\lambda^2\|v\|_2^2
=\|u\|_2^2-\frac{(u^Tv)^2}{\|v\|_2^2}.
$$

<!-- bilingual-en:start -->
Multiply by the positive number $\|v\|_2^2$ and take square roots to obtain the bound. Equality means that the residual vector $u-\lambda v$ has squared length zero, which forces every coordinate to be zero. Thus $u=\lambda v$. This also proves the equality condition instead of inferring it from a picture.
<!-- bilingual-en:end -->

乘以正数$\|v\|_2^2$并开平方，就得到所需不等式。等号意味着剩余向量$u-\lambda v$的长度平方为零，因而每个坐标都为零，也就是$u=\lambda v$。这同时证明了等号条件。

<!-- bilingual-en:start -->
## Bounding a quadratic expression
<!-- bilingual-en:end -->

控制一个二次表达式

<!-- bilingual-en:start -->
For a real square matrix $B$ and a displacement vector $h$, apply the inequality to $h$ and $Bh$. The [[谱范数|spectral norm]] bounds how much the matrix can stretch a vector, $\|Bh\|_2\le\|B\|_2\|h\|_2$. Consequently
<!-- bilingual-en:end -->

对实方阵$B$和位移向量$h$，把不等式用于$h$与$Bh$。[[谱范数]]控制矩阵能把向量拉长多少，即$\|Bh\|_2\le\|B\|_2\|h\|_2$。因此

$$|h^TBh|\le\|h\|_2\|Bh\|_2\le\|B\|_2\|h\|_2^2.$$

<!-- bilingual-en:start -->
This estimate needs no symmetry assumption on $B$. In [[多元Taylor近似|multivariable Taylor approximation]], it can turn a small matrix remainder into a bound proportional to squared displacement. For random variables, the related [[期望的Cauchy–Schwarz不等式|expectation inequality]] requires finite second moments; those probabilistic conditions should be checked separately.
<!-- bilingual-en:end -->

这个估计不要求$B$对称。在[[多元Taylor近似]]中，它可把一个很小的矩阵余量转换成与位移平方成比例的界。随机变量对应的[[期望的Cauchy–Schwarz不等式]]则要求有限二阶矩，应用时要另外检查概率条件。

## 来源与核验

<!-- bilingual-en:start -->
- [Axler, Linear Algebra Done Right, 4th ed., Theorem 6.14 and Example 6.16(a), printed p. 189 / PDF p. 203](https://linear.axler.net/LADR4e.pdf#page=203): the inequality, equality including zero vectors, and the Euclidean coordinate form.
- [Boyd–Vandenberghe, Convex Optimization, Appendix A.1.1, printed p. 633 / PDF p. 647](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=647): dot product, Euclidean norm and the bound. [Printed p. 74 / PDF p. 88](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=88) applies its squared form to Hessian estimates.
- [[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#5.4 为什么沿直线可以在参数一处取值|SOFP Lecture 1, Taylor remainder context]]: the role of the matrix norm estimate.
<!-- bilingual-en:end -->

- Axler核对定理、零向量与等号情形；Boyd–Vandenberghe核对记号和二阶估计用途。配方证明及两个坐标例子直接展开核算；原课程余项语境保留。

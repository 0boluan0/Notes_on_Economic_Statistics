---
student_os: knowledge-atom
atom_id: OPT-TOOLS-010
aliases:
  - Hessian of a quadratic form
atom_type: identity
status: source-checked
---

<!-- bilingual-en:start -->
# Hessian of a quadratic form
<!-- bilingual-en:end -->
*二次型的 Hessian*

<!-- bilingual-en:start -->
The Hessian of a quadratic form is a constant matrix that records its second derivatives. There is a factor of two to watch: if a real quadratic form is written as $q(x)=x^TSx$ with $S$ symmetric, its Hessian is $2S$, not $S$. The two matrices describe the same quadratic pattern with different normalizations.
<!-- bilingual-en:end -->

二次型的 Hessian 是记录其二阶导数的常矩阵。这里有一个容易混淆的二倍关系：实二次型若写成 $q(x)=x^TSx$，其中 $S$ 对称，那么 Hessian 是 $2S$，不是 $S$。两矩阵描述同一二次结构，但归一化方式不同。

<!-- bilingual-en:start -->
Start with the [[二次型|quadratic form]]
<!-- bilingual-en:end -->

先看这个[[二次型]]：

$$q(x_1,x_2)=3x_1^2-4x_1x_2+5x_2^2,\qquad
S=\begin{pmatrix}3&-2\\-2&5\end{pmatrix}.$$

<!-- bilingual-en:start -->
A [[偏导数|partial derivative]] changes one coordinate while holding the others fixed. Differentiating once gives the column [[全微分#实值函数、梯度与全微分|gradient]], and differentiating those entries again gives the [[Hessian矩阵|Hessian matrix]]:
<!-- bilingual-en:end -->

[[偏导数]]让一个坐标变化，同时固定其余坐标。求导一次得到列向量形式的[[全微分#实值函数、梯度与全微分|梯度]]，再对这些分量求导，得到 [[Hessian矩阵]]：

$$
\nabla q(x)=\begin{pmatrix}6x_1-4x_2\\-4x_1+10x_2\end{pmatrix},
\qquad H_q=\begin{pmatrix}6&-4\\-4&10\end{pmatrix}=2S.
$$

<!-- bilingual-en:start -->
The second derivative of $3x_1^2$ is $6$. The mixed derivative of $-4x_1x_2$ is $-4$, while each off-diagonal entry of $S$ is only $-2$: the two positions jointly encode that cross term.
<!-- bilingual-en:end -->

$3x_1^2$ 的二阶导数是 $6$。$-4x_1x_2$ 的混合偏导是 $-4$，而 $S$ 的每个非对角位置只有 $-2$：这两个位置共同记录交叉项。

> [!important] A fixed real matrix · 固定的实矩阵
> <!-- bilingual-en:start -->
> Let $n\ge1$ be an integer, let $A$ be a fixed real $n\times n$ matrix, and let $x\in\mathbb R^n$ be a variable column. With $T$ denoting [[矩阵转置|transpose]], the function $q(x)=x^TAx$ satisfies
> <!-- bilingual-en:end -->
>
> 设 $n\ge1$ 为整数，$A$ 是固定的实 $n\times n$ 矩阵，$x\in\mathbb R^n$ 是可变列向量，$T$ 表示[[矩阵转置|转置]]。函数 $q(x)=x^TAx$ 满足
>
> $$\boxed{\nabla q(x)=(A+A^T)x,\qquad H_q=A+A^T.}$$
>
> <!-- bilingual-en:start -->
> No symmetry assumption on $A$ is needed. Its [[二次型对称化|symmetric representative]] is $S=(A+A^T)/2$, so the same formulas become $\nabla q(x)=2Sx$ and $H_q=2S$. The entries are constant coefficients; a matrix depending on $x$ would require additional differentiation.
> <!-- bilingual-en:end -->
>
> 这里不要求 $A$ 对称。它的[[二次型对称化|对称代表]]是 $S=(A+A^T)/2$，因此同一公式也写成 $\nabla q(x)=2Sx$、$H_q=2S$。矩阵元素是固定系数；若矩阵本身依赖 $x$，求导就会多出其他项。

<!-- bilingual-en:start -->
## Why both copies appear
<!-- bilingual-en:end -->
*为什么出现矩阵与转置两份*

<!-- bilingual-en:start -->
Expand $q(x)=\sum_{i,j}a_{ij}x_ix_j$, where $a_{ij}$ is the entry in row $i$, column $j$. A coordinate $x_k$ can occur in the first factor or the second factor. The product rule therefore gives
<!-- bilingual-en:end -->

展开 $q(x)=\sum_{i,j}a_{ij}x_ix_j$，其中 $a_{ij}$ 是第 $i$ 行、第 $j$ 列的元素。坐标 $x_k$ 可能出现在第一个因子，也可能出现在第二个因子，因此乘积求导给出

$$
\frac{\partial q}{\partial x_k}
=\sum_j a_{kj}x_j+\sum_i a_{ik}x_i,
\qquad
\frac{\partial^2q}{\partial x_\ell\,\partial x_k}
=a_{k\ell}+a_{\ell k}.
$$

<!-- bilingual-en:start -->
The diagonal term $a_{kk}x_k^2$ correctly contributes twice. These entries form $A+A^T$, which is symmetric even if the original representation was not. For example, $\begin{pmatrix}3&-4\\0&5\end{pmatrix}$ represents the opening form and produces the same Hessian.
<!-- bilingual-en:end -->

对角项 $a_{kk}x_k^2$ 正好贡献两次。所有二阶导数排成 $A+A^T$；即使原来的表示矩阵不对称，这个结果也对称。例如，$\begin{pmatrix}3&-4\\0&5\end{pmatrix}$ 也表示开头的二次型，并给出同一个 Hessian。

<!-- bilingual-en:start -->
## The factor one half in Taylor's formula
<!-- bilingual-en:end -->
*Taylor 公式里的二分之一*

<!-- bilingual-en:start -->
For any base point $a$ and displacement $h$ in $\mathbb R^n$, the [[多元Taylor近似|second-order Taylor expression]] is exact for a quadratic form:
<!-- bilingual-en:end -->

对任意基点 $a$ 及位移 $h$，二者都在 $\mathbb R^n$ 中，二次型的[[多元Taylor近似|二阶 Taylor 表达式]]是精确等式：

$$q(a+h)=q(a)+\nabla q(a)^Th+\frac12h^TH_qh.$$

<!-- bilingual-en:start -->
There is no remainder here. Since $H_q=2S$, the last term is $h^TSh=q(h)$. In the opening example, take $a=(1,2)^T$ and $h=(1,0)^T$. The three terms are $15$, $-2$, and $3$, giving $q(2,2)=16$.
<!-- bilingual-en:end -->

这里没有余项。因为 $H_q=2S$，最后一项就是 $h^TSh=q(h)$。在开头的例子中，取 $a=(1,2)^T$、$h=(1,0)^T$，三项分别为 $15$、$-2$、$3$，所以 $q(2,2)=16$。

> [!note] Check how the function was written · 先看函数怎么写
> <!-- bilingual-en:start -->
> With symmetric $S$, writing $f(x)=\tfrac12x^TSx+b^Tx+c$ gives $H_f=S$. Writing $g(x)=x^TSx+b^Tx+c$ gives $H_g=2S$. Here $b$ is a fixed real vector and $c$ a fixed real number: linear and constant terms disappear after two derivatives.
> <!-- bilingual-en:end -->
>
> 对称 $S$ 下，若写成 $f(x)=\tfrac12x^TSx+b^Tx+c$，就有 $H_f=S$；若写成 $g(x)=x^TSx+b^Tx+c$，则 $H_g=2S$。这里 $b$ 是固定实向量，$c$ 是固定实数；一次项与常数项在求导两次后消失。
>
> <!-- bilingual-en:start -->
> The identity $h^TH_qh=2q(h)$ applies to the pure form $q$. It generally fails with the entire value of a function having extra terms: for $g(x)=x^2+3x+4$, the Hessian is $2$, but at $h=1$, $h^TH_gh=2$ whereas $2g(1)=16$.
> <!-- bilingual-en:end -->
>
> $h^TH_qh=2q(h)$ 针对纯二次型 $q$。如果函数还有其他项，一般不能把右边换成整个函数值：例如 $g(x)=x^2+3x+4$ 的 Hessian 是 $2$，但 $h=1$ 时左边为 $2$，而 $2g(1)=16$。

<!-- bilingual-en:start -->
This calculation identifies second derivatives; it does not locate an optimum by itself. For a general function, the [[Hessian 局部极小判据|positive-definite Hessian minimum test]] also requires an interior stationary point and suitable differentiability near it.
<!-- bilingual-en:end -->

以上计算识别二阶导数，本身不负责寻找最优点。对一般函数，[[Hessian 局部极小判据|正定 Hessian 的极小判据]]还要求内点驻点，以及附近适当的可微性。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[EC400 Slides Lecture 1.pdf#page=41|LSE EC400 SOFP, Lecture 1, PDF p. 41 / slide 39]]: the Hessian and second-order Taylor notation, including the factor one half. The general $A+A^T$ formula and examples above are derived directly.
- [[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#4.8 二次型的 Hessian 为什么是两倍系数矩阵|Permanent SOFP notes, Section 4.8]]: symmetric versus nonsymmetric representations and the linear/constant boundary.
- [W. Kahan, UC Berkeley Math H110, How to Recognize a Quadratic Form, p. 1](https://people.eecs.berkeley.edu/~wkahan/MathH110/qf.pdf#page=1): the symmetric representative, linear first derivative, and constant second derivative of a real quadratic form.
<!-- bilingual-en:end -->

- [[EC400 Slides Lecture 1.pdf#page=41|LSE EC400 SOFP Lecture 1，PDF 第 41 页／课件第 39 页]]：核对 Hessian、二阶 Taylor 记号及二分之一因子；一般 $A+A^T$ 公式与文内例子逐项推导。
- [[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#4.8 二次型的 Hessian 为什么是两倍系数矩阵|永久 SOFP 笔记，第 4.8 节]]：核对对称、非对称表示及一次项、常数项的边界。
- [W. Kahan，UC Berkeley Math H110，How to Recognize a Quadratic Form，第 1 页](https://people.eecs.berkeley.edu/~wkahan/MathH110/qf.pdf#page=1)：支持实二次型的对称代表、线性一阶导数和常数二阶导数。

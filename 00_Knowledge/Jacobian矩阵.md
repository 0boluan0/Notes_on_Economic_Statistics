---
aliases:
  - '向量值函数的 Jacobian 是导数线性映射的矩阵表示，并把输入微扰映成一阶输出微扰'
  - The Jacobian represents the derivative of a vector-valued function as a matrix
student_os: knowledge-atom
atom_id: CALC-MV-003
atom_set: multivariable-differentiation
atom_type: definition
status: source-checked
mastery_state: unassessed
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

# 向量值函数的 Jacobian 是导数线性映射的矩阵表示，并把输入微扰映成一阶输出微扰
<!-- bilingual-en:start -->
*The Jacobian of a vector-valued function is the matrix representation of its derivative and maps an input perturbation to its first-order output perturbation*
<!-- bilingual-en:end -->

> [!summary] 先看映射维度，再写 Jacobian
> 设 $F=(F_1,\ldots,F_m)^T:\mathbb R^n\to\mathbb R^m$ 在 $a$ 可微。在标准基下，导数 $DF(a)$ 的矩阵是
> $$
> J_F(a)=
> \begin{pmatrix}
> \dfrac{\partial F_1}{\partial x_1}&\cdots&\dfrac{\partial F_1}{\partial x_n}\\
> \vdots&\ddots&\vdots\\
> \dfrac{\partial F_m}{\partial x_1}&\cdots&\dfrac{\partial F_m}{\partial x_n}
> \end{pmatrix}_{x=a}.
> $$
> 它有 $m$ 行、$n$ 列：每一行对应一个输出分量，每一列对应一个输入坐标。
>
> <!-- bilingual-en:start -->
> For a differentiable map from $\mathbb R^n$ to $\mathbb R^m$, the displayed $m\times n$ matrix represents the derivative in the standard bases. Rows correspond to output components and columns to input coordinates. Checking these dimensions before multiplying matrices prevents many chain-rule errors.
> <!-- bilingual-en:end -->

## Jacobian 做的是局部输入—输出映射
<!-- bilingual-en:start -->
*The Jacobian is a local input-output map*
<!-- bilingual-en:end -->

可微性写成矩阵形式就是
$$
F(a+h)=F(a)+J_F(a)h+o(\|h\|).
$$
因此输入扰动 $h\in\mathbb R^n$ 经过 Jacobian 后变成一阶输出扰动 $J_F(a)h\in\mathbb R^m$。Jacobian 不是单纯把偏导排成表格；它继承了导数作为线性映射的作用方式。

<!-- bilingual-en:start -->
The matrix form of differentiability is shown above. An input perturbation in $\mathbb R^n$ is sent to a first-order output perturbation in $\mathbb R^m$. The Jacobian is therefore more than an array of partial derivatives: it acts as the local linear input-output map.
<!-- bilingual-en:end -->

例如
$$
F(x,y)=\begin{pmatrix}x^2y\\x+y\end{pmatrix},
\qquad
J_F(x,y)=\begin{pmatrix}2xy&x^2\\1&1\end{pmatrix}.
$$
在 $(1,2)$ 处，扰动 $h=(0.01,-0.02)^T$ 的一阶输出变化为
$$
J_F(1,2)h
=\begin{pmatrix}4&1\\1&1\end{pmatrix}
\begin{pmatrix}0.01\\-0.02\end{pmatrix}
=\begin{pmatrix}0.02\\-0.01\end{pmatrix}.
$$

<!-- bilingual-en:start -->
In the example, the first row combines the marginal effects on the first output and the second row does the same for the second output. Multiplication by the stated perturbation produces the two-component first-order output change shown above.
<!-- bilingual-en:end -->

## 标量函数的行导数与列梯度
<!-- bilingual-en:start -->
*The row derivative and column gradient of a scalar function*
<!-- bilingual-en:end -->

当 $m=1$ 时，Jacobian 是 $1\times n$ 行向量，而通常把梯度定义成 $n\times1$ 列向量。因此
$$
J_f(a)=Df(a)=\nabla f(a)^T.
$$
这只是转置约定，不是两个不同的微分对象。后续公式统一采用“梯度为列向量、Jacobian 的输出分量按行排列”。

<!-- bilingual-en:start -->
For a scalar-valued function, the Jacobian is a row vector while the gradient is conventionally a column vector, giving the displayed transpose relation. These are two representations of the same derivative, not two different differential objects. This note consistently uses column gradients and output-by-input Jacobians.
<!-- bilingual-en:end -->

## 行列式不是每个 Jacobian 都有的属性
<!-- bilingual-en:start -->
*Not every Jacobian has a determinant*
<!-- bilingual-en:end -->

只有 $m=n$ 时 Jacobian 才是方阵，才可以谈 [[Jacobian行列式|determinant]]。若输入维数 $k$ 小于输出维数 $n$，正则参数化的局部 $k$ 维体积改由[[矩形Jacobian体积因子|Gram 行列式平方根]]给出，而不是对矩形矩阵硬取 determinant。方阵 Jacobian 的非奇异性在 [[逆函数定理]] 和 [[隐函数定理]] 中有特殊作用，而 [[多元换元公式]] 使用其绝对值校正普通体积；这些都是建立在 Jacobian 表示之上的进一步结论，不能反过来把 Jacobian 定义成“一个行列式”。

<!-- bilingual-en:start -->
A determinant is defined only when the Jacobian is square. For a regular parametrisation from a lower-dimensional input space into a higher-dimensional output space, the local volume factor is instead the [[矩形Jacobian体积因子|square root of a Gram determinant]]. Nonsingularity in the square case has special consequences for local changes of variables and the [[隐函数定理|implicit function theorem]], but those are further theorems built on the Jacobian representation. A Jacobian itself is a matrix, not a determinant.
<!-- bilingual-en:end -->

> [!question]- 自检：矩阵应当是什么形状？
> 若 $F:\mathbb R^3\to\mathbb R^2$，$J_F(a)$ 和 $J_F(a)h$ 的维度分别是什么？
>
> <!-- bilingual-en:start -->
> If $F:\mathbb R^3\to\mathbb R^2$, what are the dimensions of $J_F(a)$ and $J_F(a)h$?
> <!-- bilingual-en:end -->
>
> **答案：** Jacobian 是 $2\times3$，乘以 $h\in\mathbb R^3$ 后得到 $\mathbb R^2$ 中的一阶输出扰动。
>
> <!-- bilingual-en:start -->
> **Answer:** The Jacobian is $2\times3$, and multiplying it by $h\in\mathbb R^3$ produces a first-order output perturbation in $\mathbb R^2$.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [MIT 18.S096, Lecture Notes and Readings](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/pages/lecture-notes-and-readings/)：直接核对导数作为线性映射、向量值函数的 Jacobian，以及标量 derivative 与 gradient 的转置关系。
- LSE EC400，*SOFP Lecture Notes*（课程讲义） p.40：交叉核对方程组中“输出方程按行、待解变量按列”的课程 Jacobian 记号；一般定义不依赖该应用。

<!-- bilingual-en:start -->
- MIT 18.S096 was checked directly for derivatives as linear maps, Jacobians of vector-valued functions, and the transpose relation between a scalar derivative and its gradient.
- EC400 SOFP Lecture Notes, p.40, was used to cross-check the course convention of arranging equations by rows and variables by columns in a system Jacobian; the general definition does not depend on that application.
<!-- bilingual-en:end -->

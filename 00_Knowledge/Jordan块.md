---
aliases:
  - "Jordan 块是对角线为同一特征值、超对角线为一且其余条目为零的方阵"
  - Jordan block
  - Jordan 块
student_os: knowledge-atom
atom_id: LA-EIG-031
atom_set: eigenvalues-linear-dynamics
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[特征对]]"
related:
  - "[[Jordan链]]"
  - "[[Jordan链向量线性无关]]"
  - "[[Jordan标准形]]"
  - "[[Jordan块的幂]]"
  - "[[Jordan块的指数]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# Jordan 块是对角线为同一特征值、超对角线为一且其余条目为零的方阵
<!-- bilingual-en:start -->
*A Jordan block has one eigenvalue on the diagonal, ones on the superdiagonal, and zeros elsewhere*
<!-- bilingual-en:end -->

> [!summary] 核心定义
> 大小为 $r$、对应特征值 $\lambda$ 的 Jordan 块是
> $$
> J_r(\lambda)=
> \begin{bmatrix}
> \lambda&1&0&\cdots&0\\
> 0&\lambda&1&\ddots&\vdots\\
> \vdots&\ddots&\ddots&\ddots&0\\
> 0&\cdots&0&\lambda&1\\
> 0&\cdots&\cdots&0&\lambda
> \end{bmatrix}
> =\lambda I+N.
> $$
> 其中 $N$ 只有超对角线为 $1$，满足 $N^r=0$；当 $r>1$ 时还有 $N^{r-1}\ne0$。
> <!-- bilingual-en:start -->
> The nilpotent part $N$ shifts one step along the block and has nilpotency index $r$.
> <!-- bilingual-en:end -->

在标准基 $e_1,\ldots,e_r$ 中，
$$
(J_r(\lambda)-\lambda I)e_1=0,
\qquad
(J_r(\lambda)-\lambda I)e_{j+1}=e_j.
$$
因此一个 Jordan 块正好对应一条长度为 $r$ 的[[Jordan链]]。对角线上的 $\lambda$ 给出标量伸缩，超对角线上的 $1$ 记录相邻层级之间的耦合。

$r=1$ 时，$J_1(\lambda)=[\lambda]$，没有幂零耦合；$r=2$ 时，
$$J_2(\lambda)=\begin{bmatrix}\lambda&1\\0&\lambda\end{bmatrix},$$
它只有一个普通特征方向，却有一个二维广义特征空间。
<!-- bilingual-en:start -->
A block of size one is already diagonal; a larger block records the failure to obtain enough ordinary eigenvectors.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> $J_3(4)-4I$ 的三次幂与二次幂分别是否为零？
>
> **答案：** 三次幂为零，二次幂不为零；它的幂零部分的指数正好是 $3$。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf|MIT 18.06SC Session 3.4 summary]]：核对 Jordan 块的矩阵结构。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.4.4 Jordan链与Jordan标准形|课程 3.4.4]]：核对 $J_r(\lambda)=\lambda I+N$ 与链基作用。

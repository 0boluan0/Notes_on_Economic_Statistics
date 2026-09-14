---
aliases:
  - "QR 分解把矩阵表示为正交矩阵与上梯形矩阵的乘积"
  - QR factorization
  - QR decomposition
student_os: knowledge-atom
atom_id: LA-PROJ-016
atom_set: orthogonal-projection-least-squares
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[标准正交组]]"
related:
  - "[[Gram-Schmidt正交化]]"
leads_to:
  - "[[薄QR分解]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
---

# QR 分解把矩阵表示为正交矩阵与上梯形矩阵的乘积
<!-- bilingual-en:start -->
*A QR factorization represents a matrix as the product of an orthogonal matrix and an upper-trapezoidal matrix*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 对 $A\in\mathbb R^{m\times n}$，完整 QR 分解写成
> $$
> A=QR,
> $$
> 其中 $Q\in\mathbb R^{m\times m}$ 是正交矩阵，$R\in\mathbb R^{m\times n}$ 是上梯形矩阵。完整分解保留输出空间的一整组标准正交基。
> <!-- bilingual-en:start -->
> For $A\in\mathbb R^{m\times n}$, a full QR factorization has the form $A=QR$, where $Q\in\mathbb R^{m\times m}$ is orthogonal and $R\in\mathbb R^{m\times n}$ is upper trapezoidal. The full factorisation keeps a complete orthonormal basis of the output space.
> <!-- bilingual-en:end -->

## 分解在做什么
<!-- bilingual-en:start -->
*What the factorization does*
<!-- bilingual-en:end -->

$Q$ 把输出空间换到一组标准正交坐标，$R$ 记录 $A$ 的列在这组坐标中的系数。上三角结构表示第 $j$ 列只使用前 $j$ 个正交方向。由于正交变换保持二范数，QR 分解尤其适合投影和最小二乘计算。
<!-- bilingual-en:start -->
The columns of $Q$ provide orthonormal output coordinates, while $R$ records the coordinates of the columns of $A$ in that basis. Its triangular structure says that column $j$ uses only the first $j$ orthonormal directions. Because orthogonal transformations preserve the Euclidean norm, QR factorization is especially useful for projection and least-squares computations.
<!-- bilingual-en:end -->

## 最小例子
<!-- bilingual-en:start -->
*Minimal example*
<!-- bilingual-en:end -->

对 $A=(3,4)^T$，可取
$$
Q=
\begin{bmatrix}
3/5&-4/5\\
4/5&3/5
\end{bmatrix},
\qquad
R=
\begin{bmatrix}5\\0\end{bmatrix}.
$$
这里 $Q^TQ=I_2$，且 $QR=A$。
<!-- bilingual-en:start -->
For $A=(3,4)^T$, the displayed orthogonal $Q$ and upper-trapezoidal $R$ satisfy $Q^TQ=I_2$ and $QR=A$.
<!-- bilingual-en:end -->

[[Gram-Schmidt正交化]]、Householder 反射和 Givens 旋转都可构造 QR，但算法实现不改变完整分解的定义。只保留前 $n$ 个正交列的独立定义、尺寸和秩边界见 [[薄QR分解]]。
<!-- bilingual-en:start -->
Gram-Schmidt, Householder reflectors, and Givens rotations can all construct QR, but the implementation does not change the definition of the full factorisation. The separate definition, dimensions, and rank boundary of retaining only the first $n$ orthonormal columns are given in [[薄QR分解|thin QR factorisation]].
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> 完整 QR 中为什么可以写 $Q^{-1}=Q^T$？
> <!-- bilingual-en:start -->
> Why may we write $Q^{-1}=Q^T$ in a full QR factorisation?
> <!-- bilingual-en:end -->
>
> **答案：** 完整分解中的 $Q$ 是方阵正交矩阵，满足 $Q^TQ=QQ^T=I_m$，所以 $Q^T$ 是双侧逆。
> <!-- bilingual-en:start -->
> **Answer:** In a full factorisation, $Q$ is square and orthogonal, so $Q^TQ=QQ^T=I_m$ and $Q^T$ is its two-sided inverse.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.4sum.pdf|MIT 18.06SC Session 2.4 summary]]：核验正交换基、Gram–Schmidt 构造 $A=QR$ 与上三角系数结构。
- [LAPACK Users’ Guide: QR Factorization](https://www.netlib.org/lapack/lug/node40.html)：核验一般 $m\times n$ 矩阵的完整 QR 形状。
<!-- bilingual-en:start -->
- The MIT 18.06SC Session 2.4 summary supports orthogonal coordinates, the Gram-Schmidt construction of $A=QR$, and the upper-triangular coefficient structure.
- The LAPACK Users’ Guide verifies the shape of a full QR factorisation for a general $m\times n$ matrix.
<!-- bilingual-en:end -->

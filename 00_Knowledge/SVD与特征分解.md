---
aliases:
- SVD versus eigendecomposition
- Singular value decomposition versus eigendecomposition
student_os: knowledge-atom
atom_id: LA-SVD-014
atom_set: singular-value-decomposition-low-rank
atom_type: distinction
status: source-checked
requires:
- '[[奇异值分解]]'
- '[[奇异向量]]'
- '[[特征对]]'
- '[[Hermitian 谱定理]]'
part_of:
- '[[奇异值分解与低秩近似.canvas]]'
related:
- '[[实对称矩阵谱定理]]'
- '[[Jordan、Schur与SVD用途边界]]'
---

<!-- bilingual-en:start -->
# SVD versus eigendecomposition
<!-- bilingual-en:end -->

*SVD 与特征分解*

<!-- bilingual-en:start -->
A matrix can be studied by asking two different questions. **Which directions does it keep on the same line?** This leads to eigenvectors. **Which input directions does it stretch into perpendicular output directions, and by how much?** This leads to the singular value decomposition (SVD). The distinction matters even for a square matrix; a rectangular matrix makes the different roles of input and output especially visible.
<!-- bilingual-en:end -->

研究一个矩阵时，可以问两个不同问题：**哪些方向经过作用后仍留在原来那条直线上？**这引出特征向量；**哪些输入方向被伸缩到彼此垂直的输出方向，伸缩多少？**这引出奇异值分解（SVD）。即使矩阵是方阵，两种问题也不同；长方矩阵则让输入与输出的不同角色更直观。

> [!example] One input direction, a different output direction / 输入与输出可以朝不同方向
> <!-- bilingual-en:start -->
> Let
> $$A=\begin{pmatrix}0&1\\0&0\end{pmatrix},\qquad e_1=(1,0)^T,\quad e_2=(0,1)^T.$$
> Then $Ae_2=e_1$: a unit input pointing vertically becomes a unit horizontal output. SVD records this as the singular pair $v_1=e_2$, $u_1=e_1$, with $\sigma_1=1$. But $e_2$ is not an eigenvector, because its output does not lie on the line through $e_2$. The only eigenvalue is zero, and its eigenvectors are nonzero multiples of $e_1$. Thus the eigenvalues are $0,0$, while the singular values are $1,0$.
> <!-- bilingual-en:end -->
>
> 取显示的矩阵与坐标向量，就有 $Ae_2=e_1$：竖直的单位输入变成水平的单位输出。SVD 把它记为奇异向量对 $v_1=e_2$、$u_1=e_1$，对应 $\sigma_1=1$。然而 $e_2$ 不是特征向量，因为输出不在经过 $e_2$ 的那条直线上。唯一特征值为零，特征向量是 $e_1$ 的非零倍数。因此特征值为 $0,0$，奇异值却为 $1,0$。

<!-- bilingual-en:start -->
## One basis for an eigenproblem; two bases for SVD
<!-- bilingual-en:end -->

*特征问题使用同一组基，SVD 分别选择两侧的基*

<!-- bilingual-en:start -->
A [[特征对|matrix eigenpair]] consists of a nonzero vector $v$ and a scalar $\lambda$ satisfying $Av=\lambda v$. It describes an endomorphism: input and output belong to the same space with the same coordinate identification. Ordinary matrix eigenvalues therefore require a square matrix. If an $n\times n$ matrix has $n$ independent eigenvectors over the chosen real or complex field, put them into the columns of $S$. The [[对角化|eigendecomposition]] is
$$A=S\Lambda S^{-1},\qquad S^{-1}AS=\Lambda.$$
Here $\Lambda$ is diagonal and contains eigenvalues, which may be negative or complex. The columns of $S$ need not be perpendicular. This is a [[相似矩阵|similarity transformation]]: use the same new basis on both sides of the map.
<!-- bilingual-en:end -->

一个[[特征对]]由非零向量 $v$ 与标量 $\lambda$ 组成，满足 $Av=\lambda v$。它描述同一空间上的映射，输入输出采用同一坐标认定，所以普通矩阵特征值要求方阵。若 $n\times n$ 矩阵在选定的实数域或复数域上有 $n$ 个线性无关的特征向量，把它们放入 $S$ 的各列，就得到显示的[[对角化|特征分解]]。对角矩阵 $\Lambda$ 包含特征值，可以有负数或复数；$S$ 的列不必垂直。这是[[相似矩阵|相似变换]]，映射两侧使用同一组新基。

<!-- bilingual-en:start -->
For an $m\times n$ real or complex matrix, [[奇异值分解|SVD]] instead gives
$$A=U\Sigma V^*,\qquad U^*AV=\Sigma.$$
The star denotes [[矩阵转置|conjugate transpose]], reducing to transpose for real matrices. In the full form, the $n$ columns of $V$ are an orthonormal input basis, while the $m$ columns of $U$ are an orthonormal output basis. “Orthonormal” means perpendicular unit vectors; see [[标准正交组|orthonormal lists and bases]]. The rectangular diagonal matrix $\Sigma$ contains nonnegative singular values. For a positive value, the paired [[奇异向量|singular vectors]] obey $Av_i=\sigma_i u_i$ and $A^*u_i=\sigma_i v_i$. The input and output directions need not coincide, even if $m=n$.
<!-- bilingual-en:end -->

对实或复 $m\times n$ 矩阵，[[奇异值分解|SVD]]则给出显示的两侧分解。星号是[[矩阵转置|共轭转置]]，实矩阵时就是转置。完整形式中，$V$ 的 $n$ 列构成输入空间的标准正交基，$U$ 的 $m$ 列构成输出空间的标准正交基；“标准正交”指彼此垂直且长度为1，见[[标准正交组]]。长方形对角矩阵 $\Sigma$ 包含非负奇异值。对于正值，配对的[[奇异向量]]满足 $Av_i=\sigma_i u_i$ 与 $A^*u_i=\sigma_i v_i$。即使 $m=n$，两侧方向也不必相同。

<!-- bilingual-en:start -->
For example, a $5\times3$ matrix sends $v\in\mathbb R^3$ to $Av\in\mathbb R^5$, so $Av=\lambda v$ does not even equate vectors of the same size. It still has an SVD. The same is true for a $3\times2$ matrix, which has no ordinary characteristic polynomial. Every finite real or complex matrix has an [[SVD存在性|SVD]], whereas the opening square matrix has only one independent eigenvector and cannot be diagonalized. Having some eigenvectors is weaker than having an eigenvector basis.
<!-- bilingual-en:end -->

例如 $5\times3$ 矩阵把 $v\in\mathbb R^3$ 送到 $Av\in\mathbb R^5$，此时 $Av=\lambda v$ 连两侧向量长度都不一致，却仍然存在 SVD。$3\times2$ 矩阵也是如此，它没有普通特征多项式。每个有限实或复矩阵都有 [[SVD存在性|SVD]]；开头的方阵却只有一个独立特征向量，不能对角化。有一些特征向量，比拥有完整特征向量基弱得多。

<!-- bilingual-en:start -->
## When the two decompositions coincide or nearly coincide
<!-- bilingual-en:end -->

*特殊结构下，两种分解如何联系*

<!-- bilingual-en:start -->
For a [[Hermitian 矩阵|Hermitian matrix]] $A=A^*$, the [[Hermitian 谱定理|spectral theorem]] supplies an orthonormal eigenbasis:
$$A=Q\Lambda Q^*.$$
If all eigenvalues are nonnegative, one can take $U=V=Q$ and $\Sigma=\Lambda$, ordering the columns consistently. If some eigenvalues are negative, their absolute values become singular values. For a nonzero eigenvalue, choose the right singular vector $v_i=q_i$ and the left vector $u_i=\operatorname{sign}(\lambda_i)q_i$, so the sign moves into the direction pair. For a zero eigenvalue, choose equal basis vectors on both sides if desired; no sign division is required. Reorder all factors together by decreasing $|\lambda_i|$.
<!-- bilingual-en:end -->

对 [[Hermitian 矩阵]] $A=A^*$，[[Hermitian 谱定理|谱定理]]提供标准正交的特征向量基，使 $A=Q\Lambda Q^*$。若全部特征值非负，可取 $U=V=Q$、$\Sigma=\Lambda$，各列保持一致排序。若含负特征值，奇异值取其绝对值。对于非零特征值，取右奇异向量 $v_i=q_i$，左奇异向量 $u_i=\operatorname{sign}(\lambda_i)q_i$，符号就进入方向对中。零特征值处可以选择两侧相同的基向量，无需除以符号；最后按 $|\lambda_i|$ 递减顺序共同重排各个因子。

> [!example] A negative eigenvalue reverses a direction / 负特征值翻转方向
> <!-- bilingual-en:start -->
> For $A=\operatorname{diag}(-2,1)$, the eigenvalues are $-2,1$, while the singular values are $2,1$. A valid SVD takes $V=I$, $U=\operatorname{diag}(-1,1)$ and $\Sigma=\operatorname{diag}(2,1)$. The positive stretch of size 2 is followed by the direction reversal in $U$.
> <!-- bilingual-en:end -->
>
> 对 $A=\operatorname{diag}(-2,1)$，特征值为 $-2,1$，奇异值为 $2,1$。取 $V=I$、$U=\operatorname{diag}(-1,1)$、$\Sigma=\operatorname{diag}(2,1)$ 就是有效 SVD：大小为2的正伸缩，配合 $U$ 中的方向翻转，恢复负作用。

<!-- bilingual-en:start -->
These are structural special cases, not a rule that singular values are always absolute eigenvalues: the opening matrix disproves that rule. For arbitrary $A$, singular values come from the nonnegative square roots of the [[奇异值与Gram谱|Gram eigenvalues of $A^*A$]], not from the eigenvalues of $A$. Use the eigenproblem to study invariant directions and repeated application of a square map; use SVD to study stretching, rank and approximation. The broader choice among [[Jordan、Schur与SVD用途边界|Jordan, Schur and SVD]] separates exact algebra from numerical eigenvalue calculations as well.
<!-- bilingual-en:end -->

这些是特殊结构带来的联系，不能推广成“奇异值总是特征值的绝对值”；开头的矩阵已给出反例。一般 $A$ 的奇异值来自 [[奇异值与Gram谱|Gram 矩阵 $A^*A$ 的非负特征值平方根]]，而非 $A$ 自身的特征值。研究方阵的不变方向与反复作用时用特征问题；研究伸缩、秩与近似时用 SVD。更广的 [[Jordan、Schur与SVD用途边界|Jordan、Schur 与 SVD 比较]]还区分精确代数与数值特征值计算。

## 来源与核验

<!-- bilingual-en:start -->
- [LAPACK Users’ Guide, SVD](https://netlib.org/lapack/lug/node32.html): rectangular dimensions, two orthonormal bases and both singular-vector pairing equations.
- [Axler, *Linear Algebra Done Right*, 4th ed., §7B and §7E](https://linear.axler.net/LADR4e.pdf#page=259): spectral theorems 7.29/7.31, Gram definition 7.65, SVD 7.70 and the comparison on printed p.274 (PDF p.288). The examples and sign-adjustment construction above are checked directly.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S05_Lecture_Lecture_29_Singular_Value_Decomposition.pdf|MIT Lecture 29 transcript]]: local two-space viewpoint and the relationship to symmetric eigendecomposition.
<!-- bilingual-en:end -->

- LAPACK SVD 说明核对长方形尺寸、两侧标准正交基及两条奇异向量配对方程。
- Axler 第7B与7E节核对谱定理7.29／7.31、Gram 定义7.65、SVD 定理7.70，以及印刷274页／PDF288页的比较。例子和符号调整构造由显示公式直接检验。
- MIT Lecture29 讲稿提供课程中的两空间视角及与对称特征分解的联系。

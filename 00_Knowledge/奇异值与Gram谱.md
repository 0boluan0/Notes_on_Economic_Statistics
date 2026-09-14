---
aliases:
  - "矩阵的正奇异值平方恰是两个 Gram 矩阵共有的正特征值"
  - Singular values and Gram spectra
  - Gram spectrum identity for singular values
student_os: knowledge-atom
atom_id: LA-SVD-025
atom_set: singular-value-decomposition-low-rank
atom_type: identity
status: source-checked
mastery_state: unassessed
requires:
  - "[[奇异值]]"
  - "[[Hermitian 谱定理]]"
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
leads_to:
  - "[[SVD手算流程]]"
  - "[[正规方程条件数平方]]"
related:
  - "[[奇异向量与Gram特征向量]]"
  - "[[Frobenius范数与奇异值]]"
---

# 矩阵的正奇异值平方恰是两个 Gram 矩阵共有的正特征值
<!-- bilingual-en:start -->
*The squared positive singular values of a matrix are exactly the positive eigenvalues shared by its two Gram matrices*
<!-- bilingual-en:end -->

> [!summary] 核心恒等式
> 若 $A=U\Sigma V^*$，则
> $$
> A^*A=V\Sigma^*\Sigma V^*,
> \qquad
> AA^*=U\Sigma\Sigma^*U^*.
> $$
> 因而 $A^*A$ 与 $AA^*$ 的正特征值相同，都是 $A$ 的正奇异值平方 $\sigma_i^2$。
> <!-- bilingual-en:start -->
> The two Gram matrices $A^*A$ and $AA^*$ have the same positive eigenvalues, namely the squared positive singular values $\sigma_i^2$ of $A$.
> <!-- bilingual-en:end -->

这条恒等式给出从 Gram 谱读取奇异值的方法：先求正特征值 $\lambda_i$，再取非负平方根 $\sigma_i=\sqrt{\lambda_i}$。它也解释了为什么奇异值适用于矩形矩阵：真正被对角化的是两个方形 Hermitian Gram 矩阵。
<!-- bilingual-en:start -->
The identity gives a constructive route from a Gram spectrum to singular values: compute each positive eigenvalue $\lambda_i$ and take its nonnegative square root $\sigma_i=\sqrt{\lambda_i}$. It also explains why singular values apply to rectangular matrices: the square Hermitian objects being diagonalised are the two Gram matrices.
<!-- bilingual-en:end -->

## 最小例子
<!-- bilingual-en:start -->
*Minimal example*
<!-- bilingual-en:end -->

对
$$
A=\begin{bmatrix}3&0\\0&1\\0&0\end{bmatrix},
$$
有 $A^*A=\operatorname{diag}(9,1)$，而 $AA^*=\operatorname{diag}(9,1,0)$。两边共有的正特征值是 $9,1$，所以正奇异值为 $3,1$。
<!-- bilingual-en:start -->
For the displayed $3\times2$ matrix, $A^*A=\operatorname{diag}(9,1)$ while $AA^*=\operatorname{diag}(9,1,0)$. Their shared positive eigenvalues are $9$ and $1$, so the positive singular values are $3$ and $1$.
<!-- bilingual-en:end -->

## 尺寸边界
<!-- bilingual-en:start -->
*Dimension boundary*
<!-- bilingual-en:end -->

只有**正谱**必须相同。$A^*A$ 是 $n\times n$，$AA^*$ 是 $m\times m$；当 $m\ne n$ 时，两者的零特征值重数可以不同。不能把“正特征值相同”误写成“包括重数在内的全部谱相同”。
<!-- bilingual-en:start -->
Only the positive spectra must agree. Because $A^*A$ is $n\times n$ and $AA^*$ is $m\times m$, their zero-eigenvalue multiplicities may differ when $m\ne n$. Equality of the positive spectra is not equality of the complete spectra with multiplicity.
<!-- bilingual-en:end -->

> [!question]- 自检
> 若 $A^*A$ 的正特征值为 $25,4$，$AA^*$ 还多出三个零特征值，$A$ 的正奇异值是什么？额外零值会改变答案吗？
>
> **答案：** 正奇异值是 $5,2$；额外零值只反映尺寸或零空间维数，不改变正奇异值。
>
> <!-- bilingual-en:start -->
> **Question:** If the positive eigenvalues of $A^*A$ are $25$ and $4$, while $AA^*$ has three additional zero eigenvalues, what are the positive singular values? Do the extra zeros change them?
>
> **Answer:** They are $5$ and $2$. The extra zeros reflect dimensions or null-space size and do not change the positive singular values.
> <!-- bilingual-en:end -->

## 来源与核验

- [LAPACK Users' Guide: Singular Value Decomposition](https://www.netlib.org/lapack/lug/node32.html)：核对 $A^*A$、$AA^*$ 的正特征值等于奇异值平方，以及矩形尺寸产生的零谱边界。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|MIT 18.06SC Session 3.5 summary]]：核对由 $A^TA$ 的特征值构造奇异值的课程路线。
<!-- bilingual-en:start -->
- The LAPACK Users' Guide supports the squared-singular-value spectra of $A^*A$ and $AA^*$ and the zero-spectrum boundary caused by rectangular dimensions.
- The MIT 18.06SC Session 3.5 summary supports constructing singular values from the eigenvalues of $A^TA$.
<!-- bilingual-en:end -->

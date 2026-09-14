---
aliases:
  - "正奇异值 σᵢ 对应的右、左奇异向量分别是 $A^*A$ 与 $AA^*$ 的 σᵢ² 特征向量"
  - Singular vectors and Gram eigenvectors
  - Gram-matrix characterisation of singular vectors
student_os: knowledge-atom
atom_id: LA-SVD-026
atom_set: singular-value-decomposition-low-rank
atom_type: equivalence
status: source-checked
mastery_state: unassessed
requires:
  - "[[奇异向量]]"
  - "[[奇异值与Gram谱]]"
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
leads_to:
  - "[[SVD手算流程]]"
related:
  - "[[SVD与四个基本子空间]]"
---

# 正奇异值 σᵢ 对应的右、左奇异向量分别是 $A^*A$ 与 $AA^*$ 的 σᵢ² 特征向量
<!-- bilingual-en:start -->
*For a positive singular value σᵢ, the corresponding right and left singular vectors are eigenvectors of $A^*A$ and $AA^*$ for eigenvalue σᵢ²*
<!-- bilingual-en:end -->

> [!summary] 谱刻画
> 若 $\sigma_i>0$ 且 $v_i,u_i$ 是一对右、左奇异向量，则
> $$
> A^*Av_i=\sigma_i^2v_i,
> \qquad
> AA^*u_i=\sigma_i^2u_i.
> $$
> 反过来，若单位向量 $v_i$ 满足 $A^*Av_i=\lambda_i v_i$ 且 $\lambda_i>0$，令 $\sigma_i=\sqrt{\lambda_i}$、$u_i=Av_i/\sigma_i$，就得到对应的奇异向量对。
> <!-- bilingual-en:start -->
> A positive singular-vector pair is a matched pair of Gram eigenvectors for eigenvalue $\sigma_i^2$. Conversely, a unit eigenvector of $A^*A$ for a positive eigenvalue generates the corresponding left singular vector by normalization.
> <!-- bilingual-en:end -->

从配对式 $Av_i=\sigma_i u_i$、$A^*u_i=\sigma_i v_i$ 再作用一次，立即得到两个 Gram 特征方程。反向构造中，
$$
\|u_i\|_2^2
=\frac{v_i^*A^*Av_i}{\sigma_i^2}=1,
$$
而 $A^*u_i=\sigma_i v_i$，所以得到的确实是单位左奇异向量。由 $AA^*$ 的正特征向量出发可作完全对称的构造。
<!-- bilingual-en:start -->
Applying $A^*$ or $A$ once more to the pairing equations gives the two Gram eigenvalue equations. In the converse direction, the displayed norm calculation shows that $u_i=Av_i/\sigma_i$ is a unit vector, and $A^*u_i=\sigma_i v_i$. Starting from a positive eigenvector of $AA^*$ gives the symmetric construction.
<!-- bilingual-en:end -->

## 最小例子
<!-- bilingual-en:start -->
*Minimal example*
<!-- bilingual-en:end -->

若 $A=\operatorname{diag}(3,1)$，则 $A^*A=AA^*=\operatorname{diag}(9,1)$。特征向量 $e_1,e_2$ 分别对应 $\sigma_1=3,\sigma_2=1$，并且 $Ae_1=3e_1$、$Ae_2=e_2$，所以左右奇异向量都可取 $e_1,e_2$。
<!-- bilingual-en:start -->
For $A=\operatorname{diag}(3,1)$, both Gram matrices equal $\operatorname{diag}(9,1)$. Their eigenvectors $e_1,e_2$ correspond to singular values $3,1$, and the left and right singular vectors may both be chosen as $e_1,e_2$.
<!-- bilingual-en:end -->

## 零谱边界
<!-- bilingual-en:start -->
*Zero-spectrum boundary*
<!-- bilingual-en:end -->

反向公式只适用于 $\lambda_i>0$，因为要除以 $\sigma_i$。零特征向量分别属于 $N(A)$ 或 $N(A^*)$；它们用于补齐完整 SVD 的两侧基，却没有由正奇异值建立的一一配对。
<!-- bilingual-en:start -->
The converse formula applies only when $\lambda_i>0$, because it divides by $\sigma_i$. Zero Gram eigenvectors belong to $N(A)$ or $N(A^*)$ and complete the two bases of a full SVD, but they are not paired by a positive singular value.
<!-- bilingual-en:end -->

> [!question]- 自检
> 已知单位向量 $v$ 满足 $A^*Av=16v$。怎样构造与它配对的单位左奇异向量？
>
> **答案：** 取 $\sigma=4$，令 $u=Av/4$；则 $\|u\|=1$ 且 $A^*u=4v$。
>
> <!-- bilingual-en:start -->
> **Question:** A unit vector $v$ satisfies $A^*Av=16v$. How do you construct its paired unit left singular vector?
>
> **Answer:** Set $\sigma=4$ and $u=Av/4$. Then $\|u\|=1$ and $A^*u=4v$.
> <!-- bilingual-en:end -->

## 来源与核验

- [LAPACK Users' Guide: Singular Value Decomposition](https://www.netlib.org/lapack/lug/node32.html)：核对左右奇异向量分别对角化 $A^*A$ 与 $AA^*$，以及正谱配对关系。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|MIT 18.06SC Session 3.5 summary]]：核对从右 Gram 特征向量构造左奇异向量的步骤。
<!-- bilingual-en:start -->
- The LAPACK Users' Guide supports the Gram-eigenvector characterisation of left and right singular vectors and their positive-spectrum pairing.
- The MIT 18.06SC Session 3.5 summary supports constructing left singular vectors from right Gram eigenvectors.
<!-- bilingual-en:end -->

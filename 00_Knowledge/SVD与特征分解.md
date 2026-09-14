---
aliases:
  - "SVD 连接输入输出空间中的两组方向而特征分解寻找同一空间中的不变方向"
  - SVD versus eigendecomposition
student_os: knowledge-atom
atom_id: LA-SVD-014
atom_set: singular-value-decomposition-low-rank
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[奇异值分解]]"
  - "[[奇异向量]]"
  - "[[特征对]]"
  - "[[Hermitian 谱定理]]"
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
related:
  - "[[实对称矩阵谱定理]]"
  - "[[Jordan、Schur与SVD用途边界]]"
---

# SVD 连接输入输出空间中的两组方向而特征分解寻找同一空间中的不变方向
<!-- bilingual-en:start -->
*The SVD links directions in the input and output spaces, whereas eigendecomposition seeks invariant directions within one space*
<!-- bilingual-en:end -->

> [!summary] 区分什么
> 特征方程 $Av=\lambda v$ 要求 $A$ 是方阵，输入与输出属于同一个空间，并寻找作用后方向不变的向量。SVD 使用
> $$
> Av_i=\sigma_i u_i,
> $$
> 允许 $A$ 是矩形矩阵；右奇异向量 $v_i$ 属于输入空间，左奇异向量 $u_i$ 属于输出空间，二者一般不是同一个向量。
> <!-- bilingual-en:start -->
> Eigenvectors are invariant directions of a square map. Singular vectors come in input-output pairs and exist for rectangular matrices as well.
> <!-- bilingual-en:end -->

例如 $A\in\mathbb R^{3\times2}$ 没有普通特征多项式，却仍有完整 SVD。即使 $A$ 是方阵，一般也不能把 $U$ 与 $V$ 合并成同一组特征向量：SVD 是双边酉变换 $U^*AV=\Sigma$，特征分解则在可对角化时是相似变换 $V^{-1}AV=\Lambda$。

两者在特殊结构下会靠近。若 $A$ 是 Hermitian 正半定矩阵，可以取 $U=V$，奇异值等于非负特征值；若 Hermitian 矩阵含负特征值，奇异值取其绝对值，而左、右奇异向量的相位还要吸收符号。这个特例不能反推一般 SVD 是特征分解的改名。

> [!question]- 自检
> 为什么一个 $5\times3$ 矩阵可以谈奇异向量，却不能直接写 $Av=\lambda v$？
>
> **答案：** $v\in\mathbb R^3$，但 $Av\in\mathbb R^5$，两者不在同一空间；SVD 用一对 $v_i\in\mathbb R^3$ 与 $u_i\in\mathbb R^5$ 连接它们。

## 来源与核验

- [LAPACK Users' Guide: Singular Value Decomposition](https://www.netlib.org/lapack/lug/node32.html)：核对矩形 SVD、左右奇异向量与尺寸。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S05_Lecture_Lecture_29_Singular_Value_Decomposition.pdf|MIT 18.06SC Lecture 29 transcript]]：核对 SVD 的两空间几何及其与对称特征分解的联系。

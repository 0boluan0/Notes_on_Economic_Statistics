---
aliases:
  - "手算 SVD 可以先对角化 Gram 矩阵，再由正奇异值配出另一侧奇异向量并补齐零空间基"
  - 奇异值分解计算流程
  - Hand computation of an SVD
student_os: knowledge-atom
atom_id: LA-SVD-021
atom_set: singular-value-decomposition-low-rank
atom_type: method
status: source-checked
mastery_state: unassessed
requires:
  - "[[SVD存在性]]"
  - "[[奇异值]]"
  - "[[奇异向量]]"
  - "[[奇异值与Gram谱]]"
  - "[[奇异向量与Gram特征向量]]"
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
related:
  - "[[SVD零块补基不唯一]]"
  - "[[SVD与四个基本子空间]]"
  - "[[正规方程条件数平方]]"
---

# 手算 SVD 可以先对角化 Gram 矩阵，再由正奇异值配出另一侧奇异向量并补齐零空间基
<!-- bilingual-en:start -->
*An SVD can be computed by diagonalising a Gram matrix, pairing the positive singular directions on the other side, and completing the nullspace bases*
<!-- bilingual-en:end -->

> [!summary] 计算顺序
> 对 $A\in\mathbb F^{m\times n}$，一种直接的手算路线是：
>
> 1. 对角化 $A^*A$，把标准正交特征向量按特征值降序排成 $V$；
> 2. 令 $p=\min(m,n)$，取前 $p$ 个特征值的非负平方根 $\sigma_i=\sqrt{\lambda_i}$，放入 $\Sigma$ 的 $p$ 个主对角位置；若 $n>m$，多出的 $n-p$ 个零特征方向属于右零空间，不会再占用 $\Sigma$ 的对角位置；
> 3. 对每个 $\sigma_i>0$，计算 $u_i=Av_i/\sigma_i$；
> 4. 若还缺左奇异向量，在 $\mathcal N(A^*)$ 中补成标准正交基；
> 5. 检查尺寸、正交性以及 $A=U\Sigma V^*$。
> <!-- bilingual-en:start -->
> Diagonalise $A^*A$, place the first $p=\min(m,n)$ square roots on the diagonal of $\Sigma$, obtain the paired left vectors from $Av_i/\sigma_i$ for positive $\sigma_i$, complete the left nullspace basis, and verify the factorisation.
> <!-- bilingual-en:end -->

若 $m<n$，也可以从较小的 $AA^*$ 出发，先求左奇异向量，再用 $v_i=A^*u_i/\sigma_i$ 配出右奇异向量。无论从哪一侧开始，都只能在 $\sigma_i>0$ 时相除；零奇异方向必须直接从相应零空间选取。

这条路线适合课程手算，却不是大规模浮点计算中推荐的数值算法。显式形成 $A^*A$ 或 $AA^*$ 会平方条件数，可能损失小奇异值的信息；数值软件通常直接对 $A$ 做双对角化等稳定算法。相同风险见 [[正规方程条件数平方]]。

例如求一个 $3\times2$ 矩阵的完整 SVD 时，$A^*A$ 只有 $2\times2$，通常先求它更省计算。得到两个右奇异向量后，正奇异值能配出相应左向量；若矩阵满列秩，还要在 $\mathbb F^3$ 中再补一个左零空间方向，才能组成 $3\times3$ 的 $U$。

最后的重构检查不可省略：特征向量次序、共同符号或复相位、零空间补基都可能使答案外观不同，但三个因子的尺寸必须正确，$U^*U=I$、$V^*V=I$，并且乘积必须回到原矩阵。

> [!question]- 自检
> 已知 $A^*Av_i=9v_i$ 且 $\|v_i\|_2=1$，怎样得到与它配对的左奇异向量？
>
> **答案：** 先取 $\sigma_i=3$，再计算 $u_i=Av_i/3$；正特征值保证这个向量是单位向量。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S05_Recitation_Problem_Solving_Computing_the_Singular_Value_Decomposition.pdf|MIT 18.06SC SVD recitation]]：核对从 $A^TA$ 求奇异值、右奇异向量和左奇异向量的手算流程。
- [LAPACK Users' Guide: Singular Value Decomposition](https://www.netlib.org/lapack/lug/node32.html)：核对因子尺寸与实复数记号。

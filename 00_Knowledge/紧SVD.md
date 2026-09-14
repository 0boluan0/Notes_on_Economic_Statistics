---
aliases:
  - "秩为 r 的矩阵的紧 SVD 只保留 r 个正奇异值及其左右奇异方向"
  - Compact singular value decomposition
student_os: knowledge-atom
atom_id: LA-SVD-013
atom_set: singular-value-decomposition-low-rank
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[奇异值分解]]"
  - "[[奇异值与秩]]"
  - "[[奇异向量]]"
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
leads_to:
  - "[[SVD秩一展开]]"
  - "[[截断SVD]]"
related:
  - "[[四个基本子空间]]"
  - "[[SVD与四个基本子空间]]"
  - "[[SVD零块补基不唯一]]"
---

# 秩为 r 的矩阵的紧 SVD 只保留 r 个正奇异值及其左右奇异方向
<!-- bilingual-en:start -->
*The compact SVD of a rank-r matrix retains only its r positive singular values and corresponding left and right singular directions*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 若 $A\in\mathbb F^{m\times n}$ 的秩为 $r$，紧 SVD 写成
> $$
> A=U_r\Sigma_rV_r^*,
> $$
> 其中 $U_r\in\mathbb F^{m\times r}$、$V_r\in\mathbb F^{n\times r}$ 的列标准正交，$\Sigma_r=\operatorname{diag}(\sigma_1,\ldots,\sigma_r)$ 且每个 $\sigma_i>0$。
> <!-- bilingual-en:start -->
> The compact SVD keeps exactly the positive singular triplets, giving factors of sizes $m\times r$, $r\times r$, and $n\times r$.
> <!-- bilingual-en:end -->

紧 SVD 仍精确重构 $A$，并不是近似。它省略的是完整 SVD 为右零空间与左零空间补齐的基向量；这些向量与四个基本子空间的完整对应见 [[SVD与四个基本子空间]]。

“紧 SVD”与“截断 SVD”不能混用。紧 SVD 只删去本来就对应零奇异值的项，仍保留全部秩 $r$；[[截断SVD]] 进一步只留前 $k<r$ 个正奇异值，因此改变矩阵并产生重构误差。

有些软件把保留 $p=\min(m,n)$ 个位置的 economy 或 thin SVD 也称为 reduced SVD。这里固定使用“紧 SVD = 只保留 $r$ 个正奇异值”的口径，避免尺寸歧义。

若 $A=0$，则 $r=0$。此时紧 SVD 可按空因子理解；实际计算通常直接把零矩阵作为单独情形处理。

> [!question]- 自检
> 一个 $7\times4$、秩为 2 的矩阵，其紧 SVD 三个因子的尺寸是什么？
>
> **答案：** $U_r$ 为 $7\times2$，$\Sigma_r$ 为 $2\times2$，$V_r$ 为 $4\times2$。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|MIT 18.06SC Session 3.5 summary]]：核对正奇异方向与四个基本子空间。
- Gilbert Strang, *Introduction to Linear Algebra*, 5th ed., §7.4：核对完整与秩约简 SVD 的尺寸和含义。

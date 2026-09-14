---
aliases:
  - "完整 SVD 在零块处分别为右零空间和左零空间选择标准正交基，两侧没有天然的一一配对"
  - 零奇异值补基不唯一
  - Nullspace basis freedom in a full SVD
student_os: knowledge-atom
atom_id: LA-SVD-010
atom_set: singular-value-decomposition-low-rank
atom_type: invariance-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[奇异向量]]"
  - "[[SVD与四个基本子空间]]"
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
related:
  - "[[紧SVD]]"
  - "[[奇异向量共同相位]]"
  - "[[重奇异值下方向不唯一]]"
---

# 完整 SVD 在零块处分别为右零空间和左零空间选择标准正交基，两侧没有天然的一一配对
<!-- bilingual-en:start -->
*At its zero block, a full SVD chooses separate orthonormal bases for the right and left nullspaces, with no natural one-to-one pairing between them*
<!-- bilingual-en:end -->

> [!summary] 不唯一在哪里
> 若 $\operatorname{rank}(A)=r$，完整 SVD 中 $v_{r+1},\ldots,v_n$ 构成 $\mathcal N(A)$ 的标准正交基，$u_{r+1},\ldots,u_m$ 构成 $\mathcal N(A^*)$ 的标准正交基。矩形零块把这些方向都乘成零，因此两侧可以分别选基，不存在与正奇异值相同的一一配对要求。
> <!-- bilingual-en:start -->
> The zero block annihilates nullspace directions, so right and left zero-subspace bases can be chosen independently.
> <!-- bilingual-en:end -->

正奇异值满足 $Av_i=\sigma_i u_i$，它把一条右方向配到一条左方向。若 $\sigma_i=0$，方程只剩 $Av_i=0$，并不能从 $v_i$ 推出某个非零 $u_i$；同理 $A^*u_j=0$ 只说明 $u_j$ 属于左零空间。矩形矩阵两侧零空间维数还可能不同，更不能强行逐列配对。

例如 $A=[1\ 0]\in\mathbb R^{1\times2}$ 的右零空间由 $(0,1)^T$ 张成，而左零空间维数为零。完整 SVD 可以补一个右奇异向量，却没有对应的额外左零方向。

紧 SVD 直接省略这些零空间补基，所以它避免了这部分表示自由；这不表示零空间不存在，只表示没有把它写进三个紧因子。

> [!question]- 自检
> 为什么零奇异值对应的左右向量不能像正奇异值那样用 $u_i=Av_i/\sigma_i$ 配对？
>
> **答案：** 因为 $\sigma_i=0$ 时不能相除，而且 $Av_i=0$ 不包含选择左零空间基的信息。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|MIT 18.06SC Session 3.5 summary]]：核对右零空间、左零空间与完整 SVD 补基的对应。
- [[四个基本子空间]]：核对两侧零空间的维数与归属。

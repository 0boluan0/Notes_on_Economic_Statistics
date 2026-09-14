---
aliases:
  - '当 $1\le k<\operatorname{rank}(A)$ 时，Frobenius 最佳秩不超过 $k$ 近似恰在 $\sigma_k>\sigma_{k+1}$ 时唯一'
  - Uniqueness of the Frobenius best low-rank approximation
student_os: knowledge-atom
atom_id: LA-SVD-012
atom_set: singular-value-decomposition-low-rank
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[截断SVD最佳低秩近似]]"
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
related:
  - "[[重奇异值下方向不唯一]]"
  - "[[谱范数低秩近似不唯一]]"
---

# 当 $1\le k<\operatorname{rank}(A)$ 时，Frobenius 最佳秩不超过 $k$ 近似恰在 $\sigma_k>\sigma_{k+1}$ 时唯一
<!-- bilingual-en:start -->
*For $1\le k<\operatorname{rank}(A)$, the Frobenius best rank-at-most-$k$ approximation is unique exactly when $\sigma_k>\sigma_{k+1}$*
<!-- bilingual-en:end -->

> [!summary] 唯一性判据
> 设 $1\le k<r=\operatorname{rank}(A)$，并在约束 $\operatorname{rank}(B)\le k$ 下最小化 $\|A-B\|_F$。最优解唯一当且仅当
> $$
> \sigma_k>\sigma_{k+1}.
> $$
> 若 $\sigma_k=\sigma_{k+1}$，截断位置切穿一个重奇异子空间，可以选择不同的 $k$ 维保留部分而维持同样的最小误差。
> <!-- bilingual-en:start -->
> The Frobenius minimiser is unique exactly when there is a strict singular-value gap at the truncation boundary.
> <!-- bilingual-en:end -->

Frobenius 误差把所有被改动方向的平方误差相加。边界有严格间隔时，最佳 $k$ 维左右子空间由前 $k$ 个奇异方向唯一确定；各向量即使改变共同相位，秩一乘积也不变，所以截断矩阵 $A_k$ 本身唯一。

若 $\sigma_k=\sigma_{k+1}$，则这个共同奇异值的子空间横跨保留端与舍弃端。在该子空间中旋转所选的 $k$ 维部分，会得到不同的截断矩阵，但尾部奇异值平方和不变，因此它们都是 Frobenius 最优解。

例如 $A=\operatorname{diag}(9,4,4,1)$、$k=2$ 时，$\sigma_2=\sigma_3$，可以在两个奇异值为 $4$ 的方向中保留任意一维，因此最优解不唯一。若奇异值改为 $9,5,4,1$，则 $\sigma_2>\sigma_3$，Frobenius 最优解唯一。

$k=0$ 时唯一候选是零矩阵；$k\ge r$ 时唯一零误差解是 $A$。这两个平凡端点不由上面的截断间隔判据描述。谱范数具有不同的唯一性行为，见 [[谱范数低秩近似不唯一]]。

> [!question]- 自检
> 奇异值为 $9,4,4,1$ 且 $k=2$ 时，Frobenius 最优解为什么不唯一？
>
> **答案：** 因为 $\sigma_2=\sigma_3$，截断切穿同一个重奇异子空间；可以选择不同的二维保留方向而保持相同尾部平方误差。

## 来源与核验

- [EPFL, Best low-rank approximation](https://sma.epfl.ch/~anchpcommon/lecture1.pdf)：核对 Frobenius 最优解唯一性的截断间隔条件。
- [Cornell CS 6210, Matrix nearness problems](https://www.cs.cornell.edu/courses/cs6210/2025fa/lec/2025-10-15.html)：核对 Frobenius 范数下的矩阵邻近问题。

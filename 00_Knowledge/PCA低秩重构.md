---
aliases:
  - "前 k 个主成分给出中心化数据的最佳秩至多 k 正交重构"
  - PCA low-rank reconstruction
  - PCA 与截断 SVD
student_os: knowledge-atom
atom_id: STAT-PCA-004
atom_set: principal-component-analysis
atom_type: optimality-result
status: source-checked
mastery_state: unassessed
part_of:
  - "[[主成分分析.canvas|主成分分析]]"
requires:
  - "[[主成分得分]]"
  - "[[低秩近似]]"
  - "[[截断SVD]]"
  - "[[Frobenius范数]]"
  - "[[截断SVD最佳低秩近似]]"
related:
  - "[[解释方差比]]"
  - "[[重根下主成分不唯一]]"
---

# 前 k 个主成分给出中心化数据的最佳秩至多 k 正交重构
<!-- bilingual-en:start -->
*The first k principal components give the best rank-at-most-k orthogonal reconstruction of centred data*
<!-- bilingual-en:end -->

> [!summary] 核心结论
> 令中心化数据矩阵 $X_c\in\mathbb R^{n\times p}$ 且 $X_c=U D V^T$。对 $0\le k\le\min(n,p)$，保留前 $k$ 个右奇异向量 $V_k$ 后，
> $$
> \widehat X_k=X_cV_kV_k^T=U_kD_kV_k^T
> $$
> 在所有秩至多为 $k$ 的矩阵中最小化 Frobenius 重构误差 $\lVert X_c-\widehat X\rVert_F$。
> <!-- bilingual-en:start -->
> Projecting centred data onto its first k right-singular directions yields the best rank-at-most-k reconstruction in squared Frobenius error.
> <!-- bilingual-en:end -->

这把“保留最多方差”和“损失最少平方重构误差”连成同一件事：丢弃部分的平方误差等于被舍弃奇异值平方之和；采用样本协方差 $S=X_c^TX_c/(n-1)$ 时，$S$ 的特征向量就是 $V$，特征值为 $d_j^2/(n-1)$。

结论依赖中心化、欧氏距离与平方损失。若使用未中心化矩阵做截断 SVD，得到的是另一种原点约束近似；若变量尺度先被改变，最小化的也是改变后坐标中的误差。非线性流形、鲁棒损失或稀疏载荷需要别的方法。

若 $k\ge\operatorname{rank}(X_c)$，重构就是 $X_c$ 本身，误差为零；“秩至多 $k$”不能偷换成“秩恰为 $k$”。

当 $1\le k<\min(n,p)$ 且截断点切穿相等的正奇异值，即 $d_k=d_{k+1}>0$，最小误差值仍然确定，但最优秩至多 $k$ 的子空间与重构可以不唯一；这与[[重根下主成分不唯一|重根处的主成分方向边界]]是同一现象。

> [!question]- 自检
> 未中心化数据直接做 truncated SVD，能否无条件称为普通 PCA？
>
> **答案：** 不能。普通 PCA 围绕样本均值分析变异；未中心化 SVD 强制低秩子空间穿过原点，目标不同。

## 来源与核验

- [scikit-learn, Decomposing signals in components](https://scikit-learn.org/stable/modules/decomposition.html#pca)：核对 PCA 先中心化、再以 SVD 获得正交成分，以及中心化矩阵的 truncated SVD 与 PCA 的对应。
- [[截断SVD最佳低秩近似]]：核对 Eckart–Young–Mirsky 最优低秩近似及误差公式。

---
aliases:
  - Low-Rank Approximation
  - truncated SVD
  - 低秩近似
  - 截断奇异值分解
tags:
  - 线性代数
  - concept
---

# Low-Rank Approximation

## 它是什么

若

$$
A=\sum_{i=1}^r\sigma_i u_iv_i^*,
\qquad \sigma_1\ge\cdots\ge\sigma_r>0,
$$

则截断到前 $k$ 项得到秩至多为 $k$ 的矩阵

$$
A_k=\sum_{i=1}^k\sigma_i u_iv_i^*.
$$

$A_k$ 保留最大的 $k$ 个输入—输出伸缩模式，舍弃较小模式。Eckart--Young 定理说明，它是在谱范数或 Frobenius 范数下最接近 $A$ 的秩至多 $k$ 矩阵。

## 误差

$$
\|A-A_k\|_2=\sigma_{k+1},
\qquad
\|A-A_k\|_F^2=\sum_{i>k}\sigma_i^2.
$$

## 边界

低秩近似保留的是能量最大的正交模式，不保证保留每个单独元素或特殊语义特征。

## 关联卡片

- [[Singular Value Decomposition]]
- [[Singular Value]]
- [[Rank-One Matrix]]
- [[PCA]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```

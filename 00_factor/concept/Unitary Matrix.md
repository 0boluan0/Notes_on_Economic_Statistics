---
aliases:
  - Unitary Matrix
  - unitary matrices
  - 酉矩阵
  - 幺正矩阵
tags:
  - 线性代数
  - concept
---

# Unitary Matrix

## 它是什么

复方阵 $Q$ 若满足

$$
Q^*Q=QQ^*=I,
$$

则称为 unitary 矩阵。它是正交矩阵在复数域中的推广，且 $Q^{-1}=Q^*$。

Unitary 变换保持 Hermitian 内积与长度：

$$
\langle Qx,Qy\rangle=x^*Q^*Qy=x^*y,
\qquad \|Qx\|=\|x\|.
$$

## 最小例子

归一化 Fourier 矩阵 $F_n/\sqrt n$ 是 unitary 矩阵。

## 边界

- 实 unitary 矩阵就是正交矩阵。
- 仅有 $Q^TQ=I$ 不足以处理复矩阵，必须使用共轭转置。

## 关联卡片

- [[Orthogonal Matrix]]
- [[Hermitian Matrix]]
- [[Normal Matrix]]
- [[Fourier Series]]

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

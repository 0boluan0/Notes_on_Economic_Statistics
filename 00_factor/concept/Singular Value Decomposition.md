---
aliases:
- Singular Value Decomposition
- SVD
- 奇异值分解
tags:
- concept
- 线性代数
---
# Singular Value Decomposition

>[!note] 它是什么
> - 「Singular Value Decomposition」是指把任意矩阵分解成两个正交矩阵和一个非负对角矩阵的乘积。
>
>[!note] 最小可检索信息
> - 定义：对任意 $m\times n$ 矩阵 A，可写成 $A=U\Sigma V^T$。
> - 符号/公式：$\Sigma$ 的对角元是奇异值，来自 $A^TA$ 的特征值平方根。
> - 最小例子：rank-1 矩阵只有一个非零奇异值。
>
## 关键性质
- SVD 适用于任意矩阵，不要求方阵也不要求可对角化。
- 它统一解释 rank、四个基本子空间、低秩逼近和伪逆。

## 关联卡片
- [[Orthogonal Matrix]]
- [[Pseudoinverse]]
- [[Diagonalization]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.outlinks, this.file.link)
)
SORT file.mtime DESC
```

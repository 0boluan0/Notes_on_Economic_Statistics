---
aliases:
- Pseudoinverse
- Moore-Penrose Pseudoinverse
- Moore Penrose inverse
- 伪逆
- 广义逆
tags:
- concept
- 线性代数
---
# Pseudoinverse

## 先记一句话

伪逆就是：**普通逆不存在时，给出最自然的 least squares / minimum norm 解的“最佳逆”**。

通常记作
$$
A^+.
$$

如果 $A$ 本来可逆，那么
$$
A^+=A^{-1}.
$$

## 它是什么

普通逆要求 $A$ 是可逆方阵。

但现实中 $A$ 可能：

- 不是方阵；
- 列不独立；
- 有非零 nullspace；
- $Ax=b$ 无精确解。

伪逆在这些情况下仍然给出统一表达：
$$
\hat{x}=A^+b.
$$

这通常代表 least squares 解；若有多个 least squares 解，伪逆选 minimum norm 的那个。

## 用 SVD 怎么看

若
$$
A=U\Sigma V^T,
$$
则
$$
A^+=V\Sigma^+U^T.
$$

$\Sigma^+$ 的做法是：

- 非零 singular value $\sigma_i$ 变成 $1/\sigma_i$；
- 零 singular value 仍然保持 0；
- 矩阵形状转置。

也就是说，能恢复的方向尽量恢复，已经被压成 0 的方向不乱补。

## 它在题里负责什么

- 表达最小二乘解：$\hat{x}=A^+b$。
- 表达最小范数解。
- 处理非方阵、秩亏矩阵。
- 连接 [[Singular Value Decomposition]]、[[Least Squares]] 和左右逆。

## 常见特例

若 $A$ 满列秩：
$$
A^+=(A^TA)^{-1}A^T.
$$

若 $A$ 满行秩：
$$
A^+=A^T(AA^T)^{-1}.
$$

这两条分别对应 left inverse 和 right inverse 的场景。

## 常见误区

- 伪逆不是随便找一个 inverse；它有明确的正交投影和最小范数意义。
- singular value 为 0 的方向不能取倒数。
- $A^+A$ 和 $AA^+$ 通常是投影矩阵，不一定是单位矩阵。

## 来自课程位置

- [[03_Positive Definite Matrices and Applications#Session 3.8 Left and right inverses and pseudoinverse|Session 3.8]]：左右逆与 Moore-Penrose pseudoinverse。
- [[03_Positive Definite Matrices and Applications#Session 3.5 Singular value decomposition|Session 3.5]]：SVD 解释伪逆结构。

## 关联卡片

- [[Singular Value Decomposition]]
- [[Least Squares]]
- [[Projection Matrix]]
- [[Left Inverse]]
- [[Right Inverse]]

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

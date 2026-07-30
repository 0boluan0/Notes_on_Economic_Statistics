---
aliases:
  - Singular Value
  - singular values
  - 奇异值
tags:
  - 线性代数
  - concept
---

# Singular Value

## 它是什么

矩阵 $A\in\mathbb F^{m\times n}$ 的奇异值是 $A^*A$ 的非负特征值的平方根：

$$
\sigma_i=\sqrt{\lambda_i(A^*A)}\ge0.
$$

它度量 $A$ 沿某个标准正交输入方向的伸缩大小。若 $v_i$ 是对应的单位特征向量且 $\sigma_i>0$，则

$$
u_i=\frac{Av_i}{\sigma_i},
\qquad Av_i=\sigma_i u_i.
$$

非零奇异值的个数等于 $\operatorname{rank}(A)$。

## 最小例子

对 $A=\operatorname{diag}(3,-2)$，$A^TA=\operatorname{diag}(9,4)$，所以奇异值为 $3,2$；奇异值不会保留负号。

## 边界

- 奇异值对任意矩形矩阵都有定义；特征值只对方阵定义。
- 零奇异值对应被 $A$ 压到零的输入方向。

## 关联卡片

- [[Singular Value Decomposition]]
- [[Matrix Rank]]
- [[Pseudoinverse]]
- [[Low-Rank Approximation]]

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

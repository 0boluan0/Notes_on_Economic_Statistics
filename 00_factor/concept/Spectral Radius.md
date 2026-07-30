---
aliases:
  - Spectral Radius
  - spectral radius of a matrix
  - 谱半径
tags:
  - 线性代数
  - concept
---

# Spectral Radius

## 它是什么

方阵 $A$ 的谱半径是其特征值模的最大值：

$$
\rho(A)=\max_{\lambda\in\sigma(A)}|\lambda|.
$$

它给出离散迭代 $u_{k+1}=Au_k$ 的基本增长尺度。若 $\rho(A)<1$，则 $A^k\to0$；若 $\rho(A)>1$，通常存在指数增长方向。

## 最小例子

$$
A=\operatorname{diag}(1,0.4,-0.2)
$$

的谱半径是 $1$。第一特征方向保持不变，其余方向衰减。

## 边界

仅知道谱半径不能描述所有瞬态或多项式因子；非正规矩阵和 Jordan 块还可能造成显著暂态增长。

## 关联卡片

- [[Eigenvalues]]
- [[Diagonalization]]
- [[Jordan Form]]
- [[Markov Matrix]]

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

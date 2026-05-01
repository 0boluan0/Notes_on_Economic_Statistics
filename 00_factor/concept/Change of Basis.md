---
aliases:
- Change of Basis
- basis change
- coordinate change
- 换基
- 基变换
tags:
- concept
- 线性代数
---
# Change of Basis

## 先记一句话

换基就是：**同一个向量或线性变换，用另一套坐标语言重新表达**。

对象没有变，坐标变了。

## 它是什么

如果基变换矩阵 $C$ 的列是新基在旧坐标下的表示，那么同一个向量满足
$$
x=C[x]_{\text{new}}.
$$

因此
$$
[x]_{\text{new}}=C^{-1}x.
$$

对线性变换矩阵，换基后通常出现
$$
B=C^{-1}AC.
$$

这就是 [[Similar Matrix]] 的来源。

## 一个最小例子

如果一组特征向量组成基底 $S$，那么换到这组基底后：
$$
S^{-1}AS=\Lambda.
$$

也就是说，原来耦合在一起的矩阵作用，在特征向量坐标里变成各方向独立缩放。

这就是 [[Diagonalization]]。

## 它在题里负责什么

- 解释为什么同一个线性变换会有不同矩阵。
- 找到让问题更简单的坐标系。
- 理解 diagonalization、Jordan form、Fourier basis、SVD 背后的共同思想。

## 常见误区

- 换基不是改变向量本身，而是改变坐标表示。
- $C$ 和 $C^{-1}$ 方向很容易写反；先写 $x=C[x]_{\text{new}}$ 最稳。
- 相似变换 $C^{-1}AC$ 只适用于同一空间到自身的线性变换。

## 来自课程位置

- [[03_Positive Definite Matrices and Applications#Session 3.7 Change of basis; image compression|Session 3.7]]：换基、图像压缩和表示变化。
- [[02_Least Squares, Determinants and Eigenvalues#Session 2.9 Diagonalization and powers of A|Session 2.9]]：换到特征向量基底计算矩阵幂。

## 关联卡片

- [[Linear Transformation]]
- [[Similar Matrix]]
- [[Diagonalization]]
- [[Spectral Decomposition]]
- [[Singular Value Decomposition]]

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

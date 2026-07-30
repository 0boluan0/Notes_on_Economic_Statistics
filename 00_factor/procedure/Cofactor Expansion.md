---
aliases:
  - Cofactor Expansion
  - Laplace expansion
  - cofactor expansion
  - 余子式展开
  - 拉普拉斯展开
tags:
  - 线性代数
  - procedure
type: procedure
---

# Cofactor Expansion

## 何时使用

计算方阵行列式，尤其当某一行或列含有许多零，或要推导递推关系时使用。

## Step 1. 选择展开行或列

优先选择非零元素最少的行或列，减少需要计算的子行列式数量。

## Step 2. 构造 minor 与 cofactor

删除第 $i$ 行、第 $j$ 列所得子矩阵记为 $M_{ij}$，代数余子式为

$$
C_{ij}=(-1)^{i+j}\det M_{ij}.
$$

符号呈棋盘格：

$$
\begin{bmatrix}+&-&+&\cdots\\-&+&-&\cdots\\+&-&+&\cdots\\\vdots&\vdots&\vdots&\ddots\end{bmatrix}.
$$

## Step 3. 展开

沿第 $i$ 行：

$$
\det A=\sum_{j=1}^n a_{ij}C_{ij}.
$$

沿第 $j$ 列：

$$
\det A=\sum_{i=1}^n a_{ij}C_{ij}.
$$

## Step 4. 递归并验算

继续计算非零项对应的低阶行列式。最后用三角化、特征值乘积或小规模直接展开交叉检查符号。

## 常见失败点

- 删除错行或错列；
- 忘记 $(-1)^{i+j}$；
- 展开一列却使用另一列的元素。

## 关联卡片

- [[Determinant]]
- [[Cramer's Rule]]
- [[Matrix Inverse]]
- [[Characteristic Polynomial]]

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

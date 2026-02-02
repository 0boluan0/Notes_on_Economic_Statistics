---
aliases:
- Linear Systems
- 线性方程组
- Linear
tags:
- concept
---
# Linear Systems

线性方程组是由若干线性方程构成的方程集合，可写为矩阵形式 $Ax=b$。

## 矩阵表示

- $A$：系数矩阵
- $x$：未知向量
- $b$：常数项向量

## 解的类型

- **唯一解**：方程组相容且主元变量数等于未知数个数。
- **无穷多解**：方程组相容且存在自由变量。
- **无解**：增广矩阵出现矛盾行。

## 关键判别（秩条件）

- 有解当且仅当 $\mathrm{rank}(A)=\mathrm{rank}(A|b)$。
- 唯一解当且仅当 $\mathrm{rank}(A)=n$（满列秩）。

## 最小例子

$$
\begin{cases}
2x+y=1\\
x-y=0
\end{cases}
\Rightarrow
A=\begin{bmatrix}2&1\\1&-1\end{bmatrix},\quad b=\begin{bmatrix}1\\0\end{bmatrix}
$$

## 相关链接

- [[00_factor/concept/Linear system solution structure|线性方程组解的结构]]
- [[00_factor/concept/Matrix Inverse|逆矩阵]]
- [[00_factor/concept/Matrix Rank|矩阵的秩]]

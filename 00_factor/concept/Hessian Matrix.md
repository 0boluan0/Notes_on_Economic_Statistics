---
aliases:
  - Hessian Matrix
  - Hessian
  - Hessian 矩阵
  - 海森矩阵
tags:
  - calculus
  - 线性代数
  - concept
---

# Hessian Matrix

## 它是什么

二阶可微标量函数 $f:\mathbb R^n\to\mathbb R$ 的 Hessian 是全部二阶偏导组成的矩阵：

$$
H_f(x)=\nabla^2 f(x)
=\left[\frac{\partial^2f}{\partial x_i\partial x_j}(x)\right]_{i,j}.
$$

当混合偏导连续时，$H_f(x)$ 为对称矩阵。它描述函数在点 $x$ 附近的二阶曲率：

$$
f(x+h)\approx f(x)+\nabla f(x)^Th+\frac12h^TH_f(x)h.
$$

## 最小例子

若 $f(x)=\frac12x^TAx-b^Tx$ 且 $A=A^T$，则

$$
\nabla f(x)=Ax-b,
\qquad H_f(x)=A.
$$

## 边界

Hessian 正定给出严格局部极小值的充分条件；半正定本身通常不足以判定。

## 关联卡片

- [[Positive Definite Matrix]]
- [[Positive Semidefinite Matrix]]
- [[Quadratic Form]]
- [[Critical Points and Extrema]]

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

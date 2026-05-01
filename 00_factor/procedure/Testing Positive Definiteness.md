---
aliases:
- Testing Positive Definiteness
- Positive Definite Test Order
- 正定矩阵判别流程
tags:
- procedure
- 线性代数
---
# Testing Positive Definiteness

## 这张卡什么时候用

题目要求你判断矩阵是否 positive definite、classify a quadratic form，或说明某个临界点是不是 minimum 时，用这张卡。

默认语境：实对称矩阵。

## 输入

- 一个实矩阵 $A$，或一个二次型 $x^TAx$。
- 题目可能给出 eigenvalues、pivots、principal minors 或具体矩阵。

## 输出

- positive definite；
- positive semidefinite；
- negative definite；
- indefinite；
- 或者说明当前判据不能直接用。

## Step 1. 先检查对称性

先看
$$
A=A^T
$$
是否成立。

若题目已经明确说 symmetric matrix，可以继续。

若不对称，不要直接套“特征值全正”“主元全正”“顺序主子式全正”这类正定判据。先回到题目要求，或改看对称部分/二次型语境。

## Step 2. 选最快的判据

- 已给 eigenvalues：直接看特征值符号。
- 矩阵维度小：看 leading principal minors。
- 正在做 elimination：看无换行交换时的 pivots。
- 给的是二次型：配方或换到特征向量坐标。

## Step 3. 执行判别

对实对称矩阵：

- 所有特征值 $>0$：positive definite。
- 所有特征值 $\geq0$ 且至少一个为 0：positive semidefinite。
- 所有特征值 $<0$：negative definite。
- 有正有负：indefinite。

也可以用：

- 所有 pivots $>0$；
- 所有 leading principal minors $>0$。

这两条给出 positive definite。

## Step 4. 写清依据

结论不要只写：

> $A$ is positive definite.

要写成：

> Since $A$ is symmetric and all eigenvalues are positive, $A$ is positive definite.

或者：

> Since all leading principal minors are positive, $A$ is positive definite.

## Step 5. 如果题目问极小值

把矩阵结论翻译回函数：

- positive definite Hessian：严格局部极小值；
- semidefinite：可能是平坦方向，不能直接说严格极小；
- indefinite：鞍点。

## 常见错误

- 只看对角线元素为正就判断正定。
- 忘记先检查对称性。
- 把 semidefinite 的 $\geq0$ 当成 positive definite 的 $>0$。
- determinant 为正就直接说正定；高维不够。

## 来自课程位置

- [[03_Positive Definite Matrices and Applications#Session 3.1 Symmetric matrices and positive definiteness|Session 3.1]]：正定矩阵的多种判据。
- [[03_Positive Definite Matrices and Applications#Session 3.3 Positive definite matrices and minima|Session 3.3]]：正定与极小值。

## 关联卡片

- [[Positive Definite Matrix]]
- [[Symmetric Matrix]]
- [[Spectral Decomposition]]
- [[Choosing Matrix Decompositions]]

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

---
aliases:
  - Computing Eigenpairs
  - eigenvalue computation
  - 求特征值与特征向量
  - 特征对计算
tags:
  - 线性代数
  - procedure
type: procedure
---

# Computing Eigenpairs

## 输入与输出

输入方阵 $A\in\mathbb F^{n\times n}$；输出全部特征值及每个特征空间的一组基。

## Step 1. 先识别结构

- 三角矩阵：特征值就是对角元；
- 对称/Hermitian：特征值为实且可选标准正交特征向量；
- 投影矩阵：特征值只可能为 $0,1$；
- Markov 矩阵：先检查 $\lambda=1$。

## Step 2. 求特征多项式

$$
p_A(\lambda)=\det(\lambda I-A).
$$

计算并因式分解 $p_A$；记录每个根的代数重数。

## Step 3. 对每个特征值解齐次系统

对每个 $\lambda$，行化简

$$
(A-\lambda I)v=0
$$

并取 $N(A-\lambda I)$ 的一组基。其维数就是几何重数。

## Step 4. 判断是否可对角化

把全部独立特征向量计数。若总数为 $n$，令这些向量组成 $S$，即可写

$$
A=S\Lambda S^{-1}.
$$

否则记录缺少的方向并转向 Jordan 结构。

## 输出检查

- 对每个候选向量直接验证 $Av=\lambda v$；
- 检查特征值之和等于 $\operatorname{tr}A$、乘积等于 $\det A$；
- 重根必须分别记录代数重数与几何重数。

## 关联卡片

- [[Characteristic Polynomial]]
- [[Algebraic and Geometric Multiplicity]]
- [[Diagonalization]]
- [[Jordan Form]]

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

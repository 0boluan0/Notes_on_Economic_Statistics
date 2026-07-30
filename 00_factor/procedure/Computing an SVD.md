---
aliases:
  - Computing an SVD
  - SVD computation
  - 计算奇异值分解
  - SVD 计算步骤
tags:
  - 线性代数
  - procedure
type: procedure
---

# Computing an SVD

## 输入与输出

输入 $A\in\mathbb F^{m\times n}$；输出 $A=U\Sigma V^*$，其中 $U,V$ unitary（实数时正交），$\Sigma$ 的对角元非负且降序排列。

## Step 1. 求 $A^*A$ 的谱

计算 Hermitian 半正定矩阵 $A^*A$ 的特征值与一组标准正交特征向量：

$$
A^*Av_i=\lambda_i v_i,
\qquad \lambda_1\ge\cdots\ge\lambda_n\ge0.
$$

## Step 2. 得到奇异值

$$
\sigma_i=\sqrt{\lambda_i}.
$$

非零奇异值数量 $r$ 应等于 $\operatorname{rank}(A)$。

## Step 3. 计算非零输出方向

对 $i\le r$，令

$$
u_i=\frac{Av_i}{\sigma_i}.
$$

这些 $u_i$ 自动标准正交。

## Step 4. 补全零奇异值方向

若需要 full SVD，把 $u_1,\ldots,u_r$ 补成 $\mathbb F^m$ 的标准正交基，并把 $v_1,\ldots,v_r$ 补成 $\mathbb F^n$ 的标准正交基。零奇异值对应的左右向量不能用 $Av_i/\sigma_i$ 计算。

## Step 5. 组装并验算

$$
U=[u_1\ \cdots\ u_m],
\quad V=[v_1\ \cdots\ v_n],
\quad \Sigma_{ii}=\sigma_i.
$$

检查 $U^*U=I$、$V^*V=I$、$Av_i=\sigma_i u_i$（$i\le r$）以及 $U\Sigma V^*=A$。

## 关联卡片

- [[Singular Value]]
- [[Singular Value Decomposition]]
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

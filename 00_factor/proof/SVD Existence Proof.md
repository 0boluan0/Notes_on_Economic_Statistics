---
aliases:
  - SVD Existence Proof
  - proof of singular value decomposition
  - 奇异值分解存在性证明
  - SVD 存在性证明
tags:
  - 线性代数
  - proof
type: proof
---

# SVD Existence Proof

## 假设与目标

给定任意 $A\in\mathbb C^{m\times n}$。目标是构造 unitary 矩阵 $U,V$ 与非负对角矩形矩阵 $\Sigma$，使

$$
A=U\Sigma V^*.
$$

## 第一步：对 $A^*A$ 使用谱定理

$A^*A$ 是 Hermitian 半正定矩阵，因为

$$
x^*A^*Ax=\|Ax\|^2\ge0.
$$

因此存在标准正交特征基 $v_1,\ldots,v_n$，满足

$$
A^*Av_i=\sigma_i^2v_i,
\qquad \sigma_i\ge0.
$$

按 $\sigma_1\ge\cdots\ge\sigma_r>0=\sigma_{r+1}=\cdots$ 排列。

## 第二步：构造非零输出方向

对 $i\le r$，定义

$$
u_i=\frac{Av_i}{\sigma_i}.
$$

其长度为

$$
\|u_i\|^2
=\frac{v_i^*A^*Av_i}{\sigma_i^2}=1.
$$

对 $i\ne j$，

$$
u_i^*u_j
=\frac{v_i^*A^*Av_j}{\sigma_i\sigma_j}
=\frac{\sigma_j^2v_i^*v_j}{\sigma_i\sigma_j}=0.
$$

所以 $u_1,\ldots,u_r$ 标准正交，且 $Av_i=\sigma_i u_i$。

## 第三步：补全正交基

把 $u_1,\ldots,u_r$ 补成 $\mathbb C^m$ 的标准正交基，并令 $U$ 以它们为列。令 $V=[v_1\ \cdots\ v_n]$。

当 $i>r$ 时，

$$
\|Av_i\|^2=v_i^*A^*Av_i=0,
$$

所以 $Av_i=0$。令 $\Sigma$ 的前 $r$ 个对角元为 $\sigma_i$，其余为零，便有

$$
AV=U\Sigma.
$$

## 结论

因为 $V$ unitary，右乘 $V^*$ 得

$$
A=U\Sigma V^*.
$$

实矩阵情形对实对称矩阵 $A^TA$ 使用实谱定理，可选择实正交的 $U,V$。

## 关联卡片

- [[Singular Value Decomposition]]
- [[Singular Value]]
- [[Spectral Theorem Proof]]
- [[Computing an SVD]]

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

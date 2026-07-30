---
aliases:
  - Spectral Theorem Proof
  - proof of the Hermitian spectral theorem
  - 谱定理证明
  - Hermitian 谱定理证明
tags:
  - 线性代数
  - proof
type: proof
---

# Spectral Theorem Proof

## 假设与目标

设 $A\in\mathbb C^{n\times n}$ 且 $A=A^*$。目标是证明存在 unitary 矩阵 $Q$ 和实对角矩阵 $\Lambda$ 使

$$
A=Q\Lambda Q^*.
$$

实对称情形是该结论在 $\mathbb R$ 上的特例。

## 第一步：特征值为实数

取特征对 $Ax=\lambda x$、$x\ne0$。由 $A=A^*$，

$$
x^*Ax=\lambda x^*x,
$$

同时

$$
x^*Ax=(Ax)^*x=\overline\lambda x^*x.
$$

由于 $x^*x>0$，有 $\lambda=\overline\lambda$，所以 $\lambda\in\mathbb R$。

## 第二步：不同特征值的特征向量正交

若 $Ax=\lambda x$、$Ay=\mu y$，则

$$
\lambda x^*y=(Ax)^*y=x^*A^*y=x^*Ay=\mu x^*y.
$$

当 $\lambda\ne\mu$ 时，$(\lambda-\mu)x^*y=0$，故 $x^*y=0$。

## 第三步：归纳构造完整正交特征基

由特征多项式在 $\mathbb C$ 上有根，$A$ 至少有一个单位特征向量 $q_1$。对任意 $y\in q_1^\perp$，

$$
q_1^*Ay=(A^*q_1)^*y=(Aq_1)^*y=\lambda_1q_1^*y=0,
$$

所以 $q_1^\perp$ 在 $A$ 下不变。$A$ 限制在该 $(n-1)$ 维子空间上仍是 Hermitian 算子。

对维数归纳，可在 $q_1^\perp$ 中找到标准正交特征基 $q_2,\ldots,q_n$。于是 $q_1,\ldots,q_n$ 是整个空间的标准正交特征基。

## 结论

令 $Q=[q_1\ \cdots\ q_n]$、$\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)$。则

$$
AQ=Q\Lambda,
\qquad Q^{-1}=Q^*,
$$

从而 $A=Q\Lambda Q^*$。

## 关联卡片

- [[Hermitian Matrix]]
- [[Unitary Matrix]]
- [[Spectral Decomposition]]
- [[Normal Matrix]]

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

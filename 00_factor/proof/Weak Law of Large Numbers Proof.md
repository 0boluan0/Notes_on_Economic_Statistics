---
aliases: [Weak Law of Large Numbers Proof, Chebyshev WLLN Proof, 弱大数定律证明]
tags: [proof, probability, discrete-mathematics]
type: proof
---
# Weak Law of Large Numbers Proof

## 假设

$X_1,X_2,\ldots$ IID，$\mathbb E[X_i]=\mu$，$\operatorname{Var}(X_i)=\sigma^2<\infty$。

## 推导

由期望线性性与独立性，
$$
\mathbb E[\bar X_n]=\mu,
\qquad
\operatorname{Var}(\bar X_n)
=\frac1{n^2}\sum_{i=1}^n\operatorname{Var}(X_i)
=\frac{\sigma^2}{n}.
$$
对任意 $\varepsilon>0$ 使用 [[Chebyshev Inequality]]：
$$
\Pr(|\bar X_n-\mu|\ge\varepsilon)
\le\frac{\sigma^2}{n\varepsilon^2}\longrightarrow0.
$$

## 结论与边界

$\bar X_n\xrightarrow{p}\mu$。该证明使用了有限方差；不能把它误称为只需有限一阶矩的一般证明。

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

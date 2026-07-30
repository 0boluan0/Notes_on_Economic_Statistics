---
aliases:
  - Rank-Nullity Theorem Proof
  - proof of rank-nullity
  - 秩-零度定理证明
tags:
  - 线性代数
  - proof
type: proof
---

# Rank-Nullity Theorem Proof

## 假设与目标

设 $T:V\to W$ 为线性映射，$V$ 有限维。目标是证明

$$
\dim V=\dim\ker T+\dim\operatorname{im}T.
$$

## 构造

取 $\ker T$ 的一组基

$$
v_1,\ldots,v_k.
$$

把它扩充为 $V$ 的一组基：

$$
v_1,\ldots,v_k,v_{k+1},\ldots,v_n.
$$

证明 $Tv_{k+1},\ldots,Tv_n$ 是 $\operatorname{im}T$ 的一组基。

## 张成性

任取 $y\in\operatorname{im}T$，存在 $x\in V$ 使 $y=Tx$。把 $x$ 在上述基中展开：

$$
x=\sum_{i=1}^n c_iv_i.
$$

因 $Tv_i=0$（$i\le k$），

$$
y=Tx=\sum_{i=k+1}^n c_iTv_i.
$$

所以 $Tv_{k+1},\ldots,Tv_n$ 张成像空间。

## 线性无关性

若

$$
\sum_{i=k+1}^n c_iTv_i=0,
$$

则

$$
T\left(\sum_{i=k+1}^n c_iv_i\right)=0.
$$

因此 $z=\sum_{i=k+1}^n c_iv_i\in\ker T$，又可写为 $z=\sum_{i=1}^k d_iv_i$。相减得

$$
\sum_{i=1}^k(-d_i)v_i+\sum_{i=k+1}^n c_iv_i=0.
$$

由于 $v_1,\ldots,v_n$ 线性无关，所有 $c_i=0$。故这些像向量线性无关。

## 结论

$Tv_{k+1},\ldots,Tv_n$ 是像空间的一组基，所以

$$
\dim\operatorname{im}T=n-k,
\qquad \dim\ker T=k.
$$

相加即得目标。

## 关联卡片

- [[Rank-Nullity Theorem]]
- [[Basis]]
- [[Linear Transformation]]

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

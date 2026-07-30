---
aliases:
  - Rank-Nullity Theorem
  - dimension theorem
  - 秩-零度定理
  - 秩零化度定理
tags:
  - 线性代数
  - concept
---

# Rank-Nullity Theorem

## 它是什么

对线性映射 $T:V\to W$，若 $V$ 有限维，则

$$
\dim V=\dim\ker T+\dim\operatorname{im}T.
$$

对 $A\in\mathbb F^{m\times n}$，这变成

$$
n=\dim N(A)+\operatorname{rank}(A).
$$

它是一条输入空间的维数账本：每个输入方向要么被保留成一个独立输出方向，要么落入零空间。

## 最小例子

若 $A$ 有 $5$ 列且秩为 $3$，则

$$
\dim N(A)=5-3=2.
$$

## 边界

- 公式中的 $n$ 是列数，因为 $N(A)$ 位于输入空间 $\mathbb F^n$。
- 左零空间使用 $A^*$：$\dim N(A^*)=m-r$。
- 完整推导见 [[Rank-Nullity Theorem Proof]]。

## 关联卡片

- [[Matrix Rank]]
- [[Null Space]]
- [[Dimension]]
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

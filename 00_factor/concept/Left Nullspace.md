---
aliases:
- Left Nullspace
- left null space
- 左零空间
tags:
- concept
- 线性代数
---
# Left Nullspace

## 它是什么
- 「Left Nullspace」是指满足 $A^Ty=0$ 的所有向量 $y$ 构成的子空间。

## 最小可检索信息
- 定义：矩阵 $A^T$ 的零空间。
- 符号/公式：$N(A^T)=\{y: A^Ty=0\}$。
- 最小例子：如果 A 的列不能铺满 $\mathbb{R}^m$，则 left nullspace 非平凡。

## 关键性质
- left nullspace 与 [[Column Space]] 正交。
- $\dim N(A^T)=m-r$，其中 $r=\operatorname{rank}(A)$。

## 关联卡片
- [[Column Space]]
- [[Row Space]]
- [[Orthogonality]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.outlinks, this.file.link)
)
SORT file.mtime DESC
```

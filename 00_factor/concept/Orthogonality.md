---
aliases:
- Orthogonality
- orthogonal
- 正交性
- 正交
tags:
- concept
- 线性代数
---
# Orthogonality

## 它是什么
- 「Orthogonality」是指两个向量或两个子空间在内积意义下互相垂直。

## 最小可检索信息
- 定义：向量 $x,y$ 正交当且仅当 $x^Ty=0$；子空间 $S,T$ 正交当且仅当任意 $s\in S,t\in T$ 都满足 $s^Tt=0$。
- 符号/公式：$x \perp y$。
- 最小例子：$(1,0)^T$ 与 $(0,1)^T$ 正交。

## 关键性质
- 正交向量组在做坐标分解和投影时最稳定。
- `row space ⟂ nullspace`，`column space ⟂ left nullspace` 是四个基本子空间的核心结构关系。

## 关联卡片
- [[Orthogonal Projection]]
- [[Orthogonal Matrix]]
- [[Row Space]]
- [[Left Nullspace]]

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

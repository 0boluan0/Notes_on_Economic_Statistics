---
aliases:
- Row Space
- 行空间
tags:
- concept
- 线性代数
---
# Row Space

>[!note] 它是什么
> - 「Row Space」是指矩阵各行向量张成的子空间，也等于 $A^T$ 的列空间。
>
>[!note] 最小可检索信息
> - 定义：由矩阵 A 的所有行向量的线性组合构成的子空间。
> - 符号/公式：$\mathrm{Row}(A)=\mathrm{Col}(A^T)$。
> - 最小例子：若两行互为倍数，则 row space 只有一维。
>
## 关键性质
- row space 与 [[Null Space]] 正交。
- row space 的维数等于矩阵的秩。

## 关联卡片
- [[Null Space]]
- [[Column Space]]
- [[Left Nullspace]]

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

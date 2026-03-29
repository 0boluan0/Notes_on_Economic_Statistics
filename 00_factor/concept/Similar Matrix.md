---
aliases:
- Similar Matrix
- Similar Matrices
- 相似矩阵
tags:
- concept
- 线性代数
---
# Similar Matrix

>[!note] 它是什么
> - 「Similar Matrix」是指表示同一线性变换、但所用基不同的两个矩阵。
>
>[!note] 最小可检索信息
> - 定义：若存在可逆矩阵 $M$ 使 $B=M^{-1}AM$，则 A 与 B 相似。
> - 符号/公式：$B=M^{-1}AM$。
> - 最小例子：对角化就是把矩阵变为与其相似的对角矩阵。
>
## 关键性质
- 相似矩阵有相同的特征值、determinant、trace。
- 相似不保留每个元素，但保留“作为线性变换”的本质。

## 关联卡片
- [[Diagonalization]]
- [[Jordan Form]]
- [[Change of Basis]]

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

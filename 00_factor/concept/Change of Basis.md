---
aliases:
- Change of Basis
- basis change
- 换基
- 基变换
tags:
- concept
- 线性代数
---
# Change of Basis

>[!note] 它是什么
> - 「Change of Basis」是指在同一向量空间中，用另一组基重新表达向量或线性变换。
>
>[!note] 最小可检索信息
> - 定义：把旧坐标转换为新坐标的过程。
> - 符号/公式：若基变换矩阵为 C，则向量坐标变成 $[x]_{\text{new}}=C^{-1}[x]_{\text{old}}$，矩阵变成 $C^{-1}AC$。
> - 最小例子：在特征向量基下，矩阵可能变成对角矩阵。
>
## 关键性质
- 换基不改变线性变换本身，只改变它的表示。
- 好的基能让问题更稀疏、更可解释或更易计算。

## 关联卡片
- [[Linear Transformation]]
- [[Similar Matrix]]
- [[Diagonalization]]

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

---
aliases:
- Orthogonal Projection
- 正交投影
tags:
- concept
- 线性代数
---
# Orthogonal Projection

>[!note] 它是什么
> - 「Orthogonal Projection」是指把一个向量映到某个子空间中距离它最近的点，并且误差向量与该子空间正交。
>
>[!note] 最小可检索信息
> - 定义：给定子空间 $S$ 与向量 $b$，其正交投影 $p\in S$ 满足 $b-p \perp S$。
> - 符号/公式：若投影到 $\operatorname{span}(a)$，则 $p=a\frac{a^Tb}{a^Ta}$。
> - 最小例子：把平面中的点垂直投到过原点的一条直线上。
>
## 关键性质
- 正交投影给出最小二乘意义下的最佳近似。
- 误差向量一定落在投影子空间的正交补中。

## 关联卡片
- [[Projection Matrix]]
- [[Least Squares]]
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

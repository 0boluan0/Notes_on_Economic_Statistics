---
aliases:
- Matrix Exponential
- 矩阵指数
tags:
- concept
- 线性代数
---
# Matrix Exponential

>[!note] 它是什么
> - 「Matrix Exponential」是指把指数函数推广到矩阵上的结果，是线性微分方程解的核心工具。
>
>[!note] 最小可检索信息
> - 定义：$e^A=\sum_{k=0}^{\infty}\dfrac{A^k}{k!}$。
> - 符号/公式：若 $u'(t)=Au(t)$，则 $u(t)=e^{At}u(0)$。
> - 最小例子：若 A 可对角化，则 $e^{At}=Se^{\Lambda t}S^{-1}$。
>
## 关键性质
- 特征值的实部决定增长与衰减。
- 对角化后计算矩阵指数最直接。

## 关联卡片
- [[Diagonalization]]
- [[Eigenvalues]]

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

---
aliases:
- Fourier Series
- 傅里叶级数
tags:
- concept
- 线性代数
---
# Fourier Series

>[!note] 它是什么
> - 「Fourier Series」是指把函数展开到一组正交三角函数基上的表达。
>
>[!note] 最小可检索信息
> - 定义：把函数写成正弦与余弦的线性组合。
> - 符号/公式：系数来自对正交基做内积投影。
> - 最小例子：周期函数可以近似为有限项三角和。
>
## 关键性质
- 本质上是“对函数空间做正交投影”。
- 它说明投影思想不仅适用于有限维向量空间。

## 关联卡片
- [[Orthogonal Projection]]
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

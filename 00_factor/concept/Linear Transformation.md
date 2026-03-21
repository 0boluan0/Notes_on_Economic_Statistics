---
aliases:
- Linear Transformation
- linear map
- 线性变换
- 线性映射
tags:
- concept
- 线性代数
---
# Linear Transformation

## 它是什么
- 「Linear Transformation」是指保持向量加法与数乘结构的映射。

## 最小可检索信息
- 定义：映射 $T$ 满足 $T(u+v)=T(u)+T(v)$，$T(cu)=cT(u)$。
- 符号/公式：给定基后，线性变换可由矩阵表示。
- 最小例子：旋转、投影、伸缩都是线性变换。

## 关键性质
- 矩阵不是第一性对象；矩阵只是线性变换在特定基下的坐标表达。
- kernel 与 image 是线性变换版本的 nullspace 与 column space。

## 关联卡片
- [[Change of Basis]]
- [[Similar Matrix]]
- [[Matrix Inverse]]

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

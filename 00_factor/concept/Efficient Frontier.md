---
aliases:
- Efficient Frontier
- 有效前沿
tags:
- concept
---
# Efficient Frontier

## 它是什么
- 「Efficient Frontier」是指在给定风险下收益最大或给定收益下风险最小的组合集合。

## 最小可检索信息
- 定义：在给定风险下收益最大或给定收益下风险最小的组合集合。
- 符号/公式：$最小方差：\min w^T\Sigma w s.t. w^T\mu=\bar\mu。$
- 最小例子：均值-方差模型的可行前沿曲线。

## 关联卡片
- [[CAPM Estimation]]
- [[Mean-Variance Portfolio Optimization]]

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

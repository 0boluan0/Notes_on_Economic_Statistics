---
aliases:
- Ordinary Least Squares Estimator
- OLS Estimator
- OLS估计量
- OLS
tags:
- concept
---
# OLS Estimator

>[!note] 它是什么
> - 「OLS Estimator」是指最小化残差平方和得到的线性回归系数估计。
>
>[!note] 最小可检索信息
> - 定义：最小化残差平方和得到的线性回归系数估计。
> - 符号/公式：$\hat\beta=(X'X)^{-1}X'y。$
> - 最小例子：估计教育对工资的影响。
>
## 关联卡片
- [[Gauss-Markov theorem]]

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

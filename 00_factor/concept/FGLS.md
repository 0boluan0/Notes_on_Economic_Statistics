---
aliases:
- Feasible Generalized Least Squares
- FGLS
- 可行广义最小二乘
tags:
- concept
---
# Feasible Generalized Least Squares (FGLS)

>[!note] 它是什么
> - 「FGLS」是指在误差协方差未知时，用估计的协方差进行GLS的估计方法。
>
>[!note] 最小可检索信息
> - 定义：在误差协方差未知时，用估计的协方差进行GLS的估计方法。
> - 符号/公式：$\hat\beta_{FGLS}=(X'\hat\Omega^{-1}X)^{-1}X'\hat\Omega^{-1}y。$
> - 最小例子：先用OLS估计异方差结构，再做GLS。
>
## 关联卡片
- [[Cochrane-Orcutt]]
- [[Heteroscedasticity Diagnosis]]
- [[White Robust Standard Errors]]

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

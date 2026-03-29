---
aliases:
- Robust Regression
- 稳健回归
tags:
- concept
---
# Robust Regression

>[!note] 它是什么
> - 「Robust Regression」是指对异常值不敏感的回归估计方法。
>
>[!note] 最小可检索信息
> - 定义：对异常值不敏感的回归估计方法。
> - 符号/公式：最小化 $\sum \rho($e_i$)$（如Huber损失）。
> - 最小例子：含离群点时用Huber回归替代OLS。
>
## 关联卡片
- [[Outlier Detection]]

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

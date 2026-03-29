---
aliases:
- Autocorrelation
- 自相关
tags:
- concept
---
# Autocorrelation

>[!note] 它是什么
> - 「Autocorrelation」是指时间序列与其滞后值之间的相关性。
>
>[!note] 最小可检索信息
> - 定义：时间序列与其滞后值之间的相关性。
> - 符号/公式：$\rho_k=\mathrm{Cov}(x_t,x_{t-k})/\mathrm{Var}(x_t)。$
> - 最小例子：AR(1)中 $\phi>0$ 产生正的一阶自相关。
>
## 关联卡片
- [[Econometrics-hub]]
- [[ADL]]
- [[Distributed Lag Model]]
- [[OLS Estimation Steps]]
- [[Durbin-Watson Statistic]]

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

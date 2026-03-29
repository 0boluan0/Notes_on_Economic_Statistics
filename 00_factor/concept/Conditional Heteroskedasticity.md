---
aliases:
- Conditional Heteroskedasticity
- 条件异方差
tags:
- concept
---
# Conditional Heteroskedasticity

>[!note] 它是什么
> - 「Conditional Heteroskedasticity」是指给定历史信息条件下方差随时间变化的现象。
>
>[!note] 最小可检索信息
> - 定义：给定历史信息条件下方差随时间变化的现象。
> - 符号/公式：$\mathrm{Var}(\varepsilon_t|\mathcal{F}_{t-1})=h_t，如 h_t=\omega+\alpha \varepsilon_{t-1}^2+\beta h_{t-1}。$
> - 最小例子：金融收益率波动在高低波动期交替。
>
## 关联卡片
- [[Volatility Modeling-hub]]
- [[ARCH]]
- [[GARCH]]
- [[GARCH Model Estimation Steps]]

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

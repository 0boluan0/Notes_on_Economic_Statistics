---
aliases:
- Kupiec Proportion of Failures Test
- Kupiec Test
- Kupiec检验
tags:
- concept
---
# Kupiec Test

>[!note] 它是什么
> - 「Kupiec Test」是指VaR回测中检验超越次数是否与名义置信水平一致。
>
>[!note] 最小可检索信息
> - 定义：VaR回测中检验超越次数是否与名义置信水平一致。
> - 符号/公式：$LR_{uc}=-2\ln\left[\frac{(1-\alpha)^{n-x}\alpha^x}{(1-\hat p)^{n-x}\hat p^x}\right]。$
> - 最小例子：250天中出现5次超越是否合理。
>
## 关联卡片
- [[VaR-hub]]
- [[00_factor/concept/Backtesting|Backtesting]]

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

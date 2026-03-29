---
aliases:
- Relative VaR
- 相对VaR
tags:
- concept
---
# Relative VaR

>[!note] 它是什么
> - 「Relative VaR」是指相对于基准或预期收益的损失分位点。
>
>[!note] 最小可检索信息
> - 定义：相对于基准或预期收益的损失分位点。
> - 符号/公式：$\mathrm{VaR}^{rel}_\alpha=Q_\alpha(\Delta V-\Delta V_{benchmark})。$
> - 最小例子：组合相对指数的1日95% VaR。
>
## 关联卡片
- [[VaR-hub]]

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

---
aliases:
- Delta-Gamma Approximation
- Delta-Gamma近似法
- Delta
tags:
- concept
---
# Delta-Gamma Approximation

>[!note] 它是什么
> - 「Delta-Gamma Approximation」是指用一阶和二阶敏感度近似组合价值变化。
>
>[!note] 最小可检索信息
> - 定义：用一阶和二阶敏感度近似组合价值变化。
> - 符号/公式：$\Delta V\approx \Delta^T\Delta x+\tfrac12\Delta x^T\Gamma\Delta x$。
> - 最小例子：期权组合在较大波动下的P&L近似。
>
## 关联卡片
- [[Option Greeks-hub]]

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

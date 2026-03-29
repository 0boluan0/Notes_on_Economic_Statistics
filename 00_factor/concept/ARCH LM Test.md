---
aliases:
- ARCH Lagrange Multiplier Test
- ARCH-LM Test
- ARCH-LM检验
- ARCH LM Test
- ARCH
tags:
- concept
---
# ARCH LM Test

>[!note] 它是什么
> - 「ARCH LM Test」是指检验残差中是否存在ARCH型条件异方差的LM检验。
>
>[!note] 最小可检索信息
> - 定义：检验残差中是否存在ARCH型条件异方差的LM检验。
> - 符号/公式：$LR=nR^2 服从 \chi^2(q) 近似分布。$
> - 最小例子：对收益率残差平方做滞后回归并检验系数显著性。
>
## 关联卡片
- [[Volatility Modeling-hub]]

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

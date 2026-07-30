---
aliases:
- Christoffersen Conditional Coverage Test
- Christoffersen Test
- Christoffersen检验
tags:
- concept
---

# Christoffersen Test

## 它是什么

「Christoffersen Test」是指VaR回测中同时检验覆盖率与独立性（条件覆盖率）的检验。

>[!note] 它是什么
> - 「Christoffersen Test」是指VaR回测中同时检验覆盖率与独立性（条件覆盖率）的检验。
>
>[!note] 最小可检索信息
> - 定义：VaR回测中同时检验覆盖率与独立性（条件覆盖率）的检验。
> - 符号/公式：$LR_{cc}=LR_{uc}+LR_{ind}$。
> - 最小例子：检验VaR例外次数是否正确且不成簇。
>
## 最小例子

检验VaR例外次数是否正确且不成簇。

## 关联卡片
- [[00_factor/concept/Backtesting|Backtesting]]
- [[Bunching]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Bunching]]。

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

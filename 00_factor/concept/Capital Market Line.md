---
aliases:
- 资本市场线
- Capital Market Line
tags:
- 金融
- concept
---

# Capital Market Line

当投资组合**在有效边界上并包含无风险资产**时，投资组合的预期收益 \mu_p 和标准差 \sigma_p 之间满足如下关系：

$$
\mu_p = r_f + \frac{\mu_m - r_f}{\sigma_m}\sigma_p
$$
- $r_f$：无风险利率
- \mu_m：市场组合的预期收益
- \sigma_m：市场组合的标准差
- \mu_p：投资组合的预期收益
- $\sigma_p$：投资组合的标准差

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

当前笔记未发现可安全确认的同主题卡片链接。

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

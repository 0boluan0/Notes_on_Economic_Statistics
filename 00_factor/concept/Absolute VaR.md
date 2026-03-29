---
aliases:
- Absolute VaR
- 绝对VaR
tags:
- concept
---
# Absolute VaR

>[!note] 它是什么
> - 「Absolute VaR」是指在给定置信水平下资产或组合损失分布的分位点（以金额/价值衡量）。
>
>[!note] 最小可检索信息
> - 定义：在给定置信水平下资产或组合损失分布的分位点（以金额/价值衡量）。
> - 符号/公式：$\mathrm{VaR}_\alpha=-Q_{\alpha}(\Delta V)。$
> - 最小例子：1日99% VaR=200万表示损失超过200万的概率为1%。
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

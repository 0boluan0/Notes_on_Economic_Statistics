---
aliases:
- Vector Autoregression
- VAR
- 向量自回归模型
- VAR Model
tags:
- concept
---
# VAR Model

>[!note] 它是什么
> - 「VAR Model」是指多变量自回归模型，用自身与其他变量的滞后来解释当前值。
>
>[!note] 最小可检索信息
> - 定义：多变量自回归模型，用自身与其他变量的滞后来解释当前值。
> - 符号/公式：$y_t=c+\sum_{i=1}^p A_i y_{t-i}+\varepsilon_t。$
> - 最小例子：用GDP与通胀的滞后预测当前值。
>
## 关联卡片
- [[Time Series Analysis-hub]]
- [[Impulse Response Function|脉冲响应函数]]
- [[Variance Decomposition|方差分解]]
- [[Johansen Cointegration Test Steps|Johansen协整检验步骤]]

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

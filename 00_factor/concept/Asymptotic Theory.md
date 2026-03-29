---
aliases:
- Asymptotic Theory
- 渐近理论
tags:
- concept
---
# Asymptotic Theory

>[!note] 它是什么
> - 「Asymptotic Theory」是指研究样本量趋于无穷时统计量的极限性质（如一致性、渐近正态性）。
>
>[!note] 最小可检索信息
> - 定义：研究样本量趋于无穷时统计量的极限性质（如一致性、渐近正态性）。
> - 符号/公式：$\sqrt{n}(\hat\theta-\theta_0) \xrightarrow{d} N(0,V)$。
> - 最小例子：样本均值的渐近分布由中心极限定理给出。
>
## 关联卡片
- [[Central Limit Theorem]]

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

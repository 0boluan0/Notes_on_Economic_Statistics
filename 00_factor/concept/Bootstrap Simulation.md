---
aliases:
- Bootstrap Simulation
- Bootstrap Method
- 自助法模拟
tags:
- concept
---
# Bootstrap Simulation

>[!note] 它是什么
> - 「Bootstrap Simulation」是指通过对样本进行有放回重抽样来近似统计量分布的方法。
>
>[!note] 最小可检索信息
> - 定义：通过对样本进行有放回重抽样来近似统计量分布的方法。
> - 符号/公式：从样本 $\{$x_i$\}$ 抽取 $B$ 次重样本并计算统计量 $T^*$。
> - 最小例子：用bootstrap估计均值的95%置信区间。
>
## 关联卡片
- [[Historical Simulation Method]]

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

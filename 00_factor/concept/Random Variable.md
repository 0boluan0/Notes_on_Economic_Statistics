---
aliases: [Random Variable, 随机变量]
tags: [concept, discrete-mathematics, probability]
---
# Random Variable

随机变量是从样本空间到数值集合的函数 $X:\Omega\to\mathbb R$。随机的是输入 outcome；$X$ 本身是确定函数。

事件 $X=x$ 表示原像 $\{\omega:X(\omega)=x\}$。随机变量可以把复杂 outcomes 压缩成题目关心的数值，但不同 outcomes 可能给出同一个值。

分布见 [[Probability Mass Function]]，均值见 [[Expected Value]]。

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

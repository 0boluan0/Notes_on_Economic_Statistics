---
aliases: [Linearity of Expectation, 期望线性性]
tags: [concept, discrete-mathematics, probability]
---
# Linearity of Expectation

只要各项期望存在，任意有限个随机变量和常数满足
$$
\mathbb E\left[\sum_i a_iX_i+c\right]
=\sum_i a_i\mathbb E[X_i]+c.
$$

这一结论不要求 $X_i$ 独立。对可数无穷和交换求和与期望还需要非负性、绝对可积等附加条件，不能直接套用有限和公式。独立通常只在计算乘积期望或方差可加性时需要。

配合 [[Indicator Random Variable]] 可把“对象个数”的期望拆成各对象被计入的概率之和。

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

---
aliases: [Probability Mass Function, PMF, 概率质量函数]
tags: [concept, discrete-mathematics, probability]
---
# Probability Mass Function

离散随机变量 $X$ 的 PMF 为
$$
p_X(x)=\Pr(X=x).
$$
它满足 $p_X(x)\ge0$ 且 $\sum_xp_X(x)=1$。区间概率通过相应取值求和获得。

MIT 6.042J 的部分材料把离散分布函数称作 density；本库使用 PMF，避免与连续型 probability density function 混淆。

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

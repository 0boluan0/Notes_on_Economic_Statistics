---
aliases: [Chebyshev Inequality, Chebyshev Bound, 切比雪夫不等式]
tags: [concept, discrete-mathematics, probability]
---
# Chebyshev Inequality

若 $X$ 有均值 $\mu$ 与有限方差 $\sigma^2$，则对 $a>0$，
$$
\Pr(|X-\mu|\ge a)\le\frac{\sigma^2}{a^2}.
$$
等价地，$\Pr(|X-\mu|\ge k\sigma)\le1/k^2$。

它不要求分布形状，但只控制围绕均值的双侧偏离；需要更强尾界时通常必须加入独立性或分布假设。

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

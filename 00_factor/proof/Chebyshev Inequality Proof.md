---
aliases: [Chebyshev Inequality Proof, 切比雪夫不等式证明]
tags: [proof, discrete-mathematics, probability]
type: proof
---
# Chebyshev Inequality Proof

令非负随机变量 $Y=(X-\mu)^2$。事件
$$
|X-\mu|\ge a
$$
等价于 $Y\ge a^2$。对 $Y$ 使用 [[Markov Inequality]]：
$$
\Pr(|X-\mu|\ge a)
=\Pr(Y\ge a^2)
\le\frac{\mathbb E[Y]}{a^2}
=\frac{\operatorname{Var}(X)}{a^2}.
$$

有限方差和 $a>0$ 是必需条件。

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

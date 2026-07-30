---
aliases: [Natural Logarithm, Natural Log, 自然对数, ln]
tags: [concept, calculus]
---
# Natural Logarithm

自然对数 $\ln x$ 是 $e^x$ 在正实数上的反函数，也可定义为

$$
\ln x=\int_1^x\frac{1}{t}\,dt,\qquad x>0.
$$

因此 $(\ln x)'=1/x$，并有 $\ln(xy)=\ln x+\ln y$。

## 最小例子

$\ln e=1$，且 $\ln(e^x)=x$。

## 关联卡片

- [[Exponential Function]]
- [[Inverse Function Derivative]]
- [[Fundamental Theorem of Calculus]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```

---
aliases:
- 两个随机变量线性组合
- Linear Combination of Two Random Variables
tags:
  - 数学
  - concept
---
# Linear Combination of Two Random Variables

假设有两个随机变量 $X$ 和 $Y$，它们的方差分别为 $\mathrm{Var}(X)$ 和 $\mathrm{Var}(Y)$，协方差为 $\mathrm{Cov}(X, Y)$。考虑它们的线性组合：

$$
Z = aX + bY
$$

其中 $a$ 和 $b$ 是常数。那么 $Z$ 的方差公式为：
$$
\mathrm{Var}(Z) = a^2 \mathrm{Var}(X) + b^2 \mathrm{Var}(Y) + 2ab\mathrm{Cov}(X, Y)
$$

如果 $X$ 和 $Y$ **独立**，那么 $\mathrm{Cov}(X, Y) = 0$，公式简化为：
$$
\mathrm{Var}(Z) = a^2 \mathrm{Var}(X) + b^2 \mathrm{Var}(Y)
$$


## 最小例子

把 **Linear Combination of Two Random Variables** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
## 概念边界

本卡片只回答“它是什么”，集中在定义、核心关系与最小例子；具体估计步骤、证明和诊断流程应分别放在 procedure、proof 或 system 卡片中。
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

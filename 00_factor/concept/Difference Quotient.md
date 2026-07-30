---
aliases:
- difference quotient
- Difference Quotient
- 差商
tags:
- math/calculus
- concept
---

# Difference Quotient

## 它是什么

当 $\Delta x\to 0$ 时，割线趋近于切线，差商的极限就是 [[Derivative|导数]]：

Difference quotient（差商）是用两个点之间的函数值变化量除以自变量变化量，表示函数在一段区间上的**平均变化率**。

> [!note] 定义
> 对函数 $f(x)$，在 $x_0$ 附近取一个小变化量 $\Delta x$，差商定义为：
>
> $$
> \frac{\Delta f}{\Delta x}
> =
> \frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}
> $$
>
> 其中 $\Delta x\neq 0$。

## 最小理解

- 分子 $\Delta f=f(x_0+\Delta x)-f(x_0)$ 表示函数值变化。
- 分母 $\Delta x$ 表示输入变化。
- 差商表示从 $x_0$ 到 $x_0+\Delta x$ 这段上的平均变化率。

## 几何意义

差商是连接两点

$$
(x_0,f(x_0)),\qquad (x_0+\Delta x,f(x_0+\Delta x))
$$

的割线斜率。

当 $\Delta x\to 0$ 时，割线趋近于切线，差商的极限就是 [[Derivative|导数]]：

$$
f'(x_0)
=
\lim_{\Delta x\to 0}
\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}
$$

## 注意点

不能一开始就令 $\Delta x=0$。直接代入通常会得到 $\frac{0}{0}$，这不是导数，而是未定式。正确步骤是：

1. 保持 $\Delta x\neq 0$。
2. 化简差商。
3. 再取极限 $\Delta x\to 0$。

## 相关链接

- [[Derivative|导数]]
- [[Limit|极限]]
- [[geometric interpretation of derivative|导数的几何意义]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Derivative]]、[[Limit]]、[[geometric interpretation of derivative]]。

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

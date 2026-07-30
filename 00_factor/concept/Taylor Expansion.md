---
aliases:
- Taylor Expansion
- Taylor Formula
- Taylor series
- 泰勒公式
tags:
- concept
- calculus
---
# Taylor Expansion

## 一句话记忆

Taylor expansion 用函数在展开点处的导数，构造其局部多项式近似。

## 它是什么

若函数在 $a$ 附近足够光滑，其 $n$ 阶 Taylor polynomial 为

$$
T_n(x)=\sum_{k=0}^{n}\frac{f^{(k)}(a)}{k!}(x-a)^k.
$$

保留有限项时是局部近似；无穷级数收敛到原函数时才可称为 Taylor series。截断误差可由余项表示，例如 Lagrange 余项为

$$
R_n(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-a)^{n+1},
$$

其中 $\xi$ 位于 $a$ 与 $x$ 之间。

## 最小例子

在 $a=0$ 处，$e^x\approx1+x+\frac{x^2}{2}$。目标点越接近展开点、保留项数越多，通常近似越好，但仍需注意收敛范围。

## 易混点

- Taylor **多项式**是有限项近似；Taylor **级数**是无穷项表达。
- 它是局部近似工具，不自动保证远离 $a$ 时仍准确。
>
## 关联卡片
- [[Lagrange Mean Value Theorem]]
- 余项与 Maclaurin 展开属于同一 Taylor 展开框架；本仓库暂未单独建立对应卡片。

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

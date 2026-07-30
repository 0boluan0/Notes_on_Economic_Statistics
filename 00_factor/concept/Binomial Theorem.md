---
aliases:
- Binomial Theorem
- binomial theorem
- 二项式定理
- 二项式展开
tags:
- math/algebra
- math/calculus
- discrete-mathematics
- counting
- concept
---

# Binomial Theorem

Binomial theorem（二项式定理）描述 $(a+b)^n$ 展开后各项的系数和幂次结构。

> [!note] 定义
> 对非负整数 $n$：
>
> $$
> (a+b)^n
> =
> \sum_{k=0}^{n}\binom{n}{k}a^{n-k}b^k
> $$
>
> 其中
>
> $$
> \binom{n}{k}=\frac{n!}{k!(n-k)!}
> $$

## 最小理解

展开 $(a+b)^n$ 时，每一项都从 $n$ 个因子里选出若干个 $b$，其余位置取 $a$。$\binom{n}{k}$ 表示从 $n$ 个位置中选 $k$ 个位置放 $b$ 的方法数。

例如：

$$
(a+b)^3
=a^3+3a^2b+3ab^2+b^3
$$

## 在导数推导中的作用

在用 [[Difference Quotient|差商]] 推导幂函数导数时，需要展开：

$$
(x+\Delta x)^n
$$

由二项式定理：

$$
(x+\Delta x)^n
=x^n+nx^{n-1}\Delta x+\binom{n}{2}x^{n-2}(\Delta x)^2+\cdots+(\Delta x)^n
$$

求导推导里常把二阶及以上的 $\Delta x$ 项合并写成：

$$
(x+\Delta x)^n
=x^n+nx^{n-1}\Delta x+O((\Delta x)^2)
$$

这个 $O((\Delta x)^2)$ 是在 $x$ 固定、$\Delta x\to 0$ 时使用的简写。

代入差商：

$$
\frac{(x+\Delta x)^n-x^n}{\Delta x}
$$

$x^n$ 会抵消，剩下主导项 $nx^{n-1}$；所有高阶项在 $\Delta x\to 0$ 时消失。因此：

$$
\frac{d}{dx}x^n=nx^{n-1}
$$

## 相关链接

- [[Derivative|导数]]
- [[Difference Quotient|差商]]
- [[Bijective Counting Principle|双射计数原理]]
- [[Rule of Sum and Product|加法与乘法计数法则]]
- [[Stars and Bars|隔板法]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Difference Quotient]]、[[Derivative]]。

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

---
aliases:
- 求原函数
- 积分
- Integral
tags:
- 数学
- 微积分
- concept
---
积分是导数的逆运算，用于求函数的原函数或计算函数曲线下的面积。

## 不定积分

不定积分是求函数的所有原函数。

$\int f(x) dx = F(x) + C$

其中 $F'(x) = f(x)$，C 是任意常数。

## 定积分

定积分计算函数在某区间内的定值。

$\int_a^b f(x) dx = F(b) - F(a)$

## 定积分的几何意义

定积分表示函数曲线与x轴之间的有向面积。

## 基本积分公式

1. $\int k dx = kx + C（k为常数）$
2. $\int x^n dx = \frac{x^{n+1}}{n+1} + C（n ≠ -1）$
3. $\int \frac{1}{x} dx = \ln|x| + C$
4. $\int e^x dx = e^x + C$
5. $\int a^x dx = \frac{a^x}{\ln a} + C$
6. $\int \sin x dx = -\cos x + C$
7. $\int \cos x dx = \sin x + C$

## 积分性质

1. **线性性质**：$\int [af(x) + bg(x)]dx = a\int f(x)dx + b\int g(x)dx$
2. **区间可加性**：$\int_a^b f(x)dx = \int_a^c f(x)dx + \int_c^b f(x)dx$
3. **牛顿-莱布尼茨公式**：$\int_a^b f(x)dx = F(b) - F(a)$

## 换元积分法

### 第一换元法（凑微分）

$\int f(g(x))g'(x)dx = \int f(u)du，其中 u = g(x)$

### 第二换元法（变量替换）

$设 x = \phi(t)，则 \int f(x)dx = \int f(\phi(t))\phi'(t)dt$

## 分部积分法

$\int u dv = uv - \int v du$

## 应用

1. **求面积**：定积分计算曲线下面积
2. **求体积**：旋转体体积
3. **求弧长**：曲线弧长
4. **求平均值**：函数在区间内的平均值
5. **经济学应用**：消费者剩余、生产者剩余

## 相关链接
[[Derivative (2)|导数]]
[[Fundamental Theorem of Calculus|微积分基本定理]]

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

---
aliases:
- Derivative
- derivative
- 导数
- 微分
tags:
- math/calculus
- concept
---

# Derivative

导数是描述函数在某一点处**瞬时变化率**的数学概念，也是微积分中把“局部变化”精确化的核心工具。

> [!note] 定义
> 函数 $f(x)$ 在点 $x_0$ 处的导数定义为：
>
> $$
> f'(x_0)=\lim_{h\to 0}\frac{f(x_0+h)-f(x_0)}{h}
> $$
>
> 如果这个极限存在，则称 $f$ 在 $x_0$ 处可导。

## 最小理解

导数来自 [[Difference Quotient|差商]] 的极限：

$$
\frac{\Delta f}{\Delta x}
=\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}
$$

- $\frac{\Delta f}{\Delta x}$：[[Difference Quotient|差商]]，表示平均变化率。
- $\lim_{\Delta x\to 0}\frac{\Delta f}{\Delta x}$：瞬时变化率。
- 直接令 $\Delta x=0$ 通常会得到 $\frac{0}{0}$，所以必须先化简差商，再取极限。

## 几何意义

导数表示函数图像在某点处的切线斜率：

$$
f'(x_0)=\text{slope of tangent line at }x_0
$$

切线方程为：

$$
y-f(x_0)=f'(x_0)(x-x_0)
$$

Related: [[geometric interpretation of derivative|导数的几何意义]]。

## 物理意义

导数表示瞬时变化率：

- 位移 $s$ 对时间 $t$ 的导数 $\frac{ds}{dt}$ 是速度。
- 速度对时间的导数是加速度。
- 电荷 $q$ 对时间的导数 $\frac{dq}{dt}$ 是电流。
- 温度 $T$ 对位置 $x$ 的导数 $\frac{dT}{dx}$ 是温度梯度。

## 常用记号

如果 $y=f(x)$，导数可以写作：

$$
f'(x),\qquad \frac{df}{dx},\qquad \frac{dy}{dx},\qquad Df
$$

$f'(x_0)$ 强调在点 $x_0$ 的导数值；$\frac{dy}{dx}$ 强调 $y$ 相对 $x$ 的变化率。

## 基本求导规则

| 规则 | 公式 |
|---|---:|
| 常数函数 | $(C)'=0$ |
| 幂函数 | $(x^r)'=rx^{r-1}$ |
| 和 | $(u+v)'=u'+v'$ |
| 常数倍 | $(cu)'=cu'$ |
| 积 | $(uv)'=u'v+uv'$ |
| 商 | $\left(\frac{u}{v}\right)'=\frac{u'v-uv'}{v^2}$ |
| 链式法则 | $\frac{d}{dx}f(u(x))=f'(u(x))u'(x)$ |

常见函数：

| 函数 | 导数 |
|---:|---:|
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |
| $\tan x$ | $\sec^2 x$ |
| $e^x$ | $e^x$ |
| $a^x$ | $(\ln a)a^x$ |
| $\ln x$ | $\frac{1}{x}$ |

## 可导性与连续性

- 可导 $\Rightarrow$ 连续。
- 连续 $\nRightarrow$ 可导。

直观上，可导要求函数在该点不仅不断开，而且局部要有稳定的切线斜率；尖点、折点、竖直切线处可能连续但不可导。

## 应用

- **切线与线性近似**：用 $f'(x_0)$ 写出局部线性模型。
- **变化率问题**：速度、增长率、敏感性分析。
- **极值与单调性**：用导数符号判断增减与候选极值点。
- **凹凸性**：用二阶导数判断曲率。
- **经济学边际量**：边际成本、边际收益、边际效用等。

## 相关链接

- [[Limit|极限]]
- [[Integral|积分]]
- [[geometric interpretation of derivative|导数的几何意义]]

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

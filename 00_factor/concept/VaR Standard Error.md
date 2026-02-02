---
aliases:
- VaR标准误
- VaR
- VaR Standard Error
tags:
- concept
---
**VaR标准误差的公式**
VaR的标准误差（以分位数估计的标准误为例）：

$$
\text{SE}(\text{VaR}) = \frac{ \sqrt{ p(1-p) } }{ n, f(\text{VaR}) }
$$
实际上更常见的形式是：
$$
\text{SE}(\text{VaR}) = \frac{1}{ f(\text{VaR}) } \cdot \sqrt{ \frac{ p(1-p) }{ n } }
$$

- $p$：VaR的置信水平（此处$0.95$）
- $n$：样本量（此处$1000$）
- $f(\text{VaR})$：在VaR处的概率密度函数值（此处$0.01$）

==这个$f(\text{VaR})$通常会给,但是如果题目中说了正态,就是让你自己算:==

正态分布密度函数：

$$
f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x-\mu)^2}{2\sigma^2} \right)
$$

## 相关链接

- 基础概念：[[VaR]]
- 应用：用于评估[[VaR]]估计的精度，以及[[00_factor/system/Backtesting|回溯检验]]




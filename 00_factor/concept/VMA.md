---
aliases:
- VMA
- Vector Moving Average
- Vector MA
- 向量移动平均
- 向量移动平均表示
tags:
- concept
- 时间序列
---
# VMA

## 先记一句话

VMA 是：**把 VAR 中的当前变量写成当前和过去冲击的无限加权和**。

## 它是什么

稳定 VAR 可以写成：
$$
x_t=\mu+\sum_{i=0}^{\infty}\Phi(i)e_{t-i}.
$$

如果进一步使用结构冲击 $\varepsilon_t$，则：
$$
x_t=\mu+\sum_{i=0}^{\infty}\Psi(i)\varepsilon_{t-i}.
$$

## 它解决什么判断

VMA 是做以下事情的桥梁：

- 推导 [[Impulse Response Function]]；
- 分析冲击如何跨期传导；
- 做 [[Variance Decomposition]]；
- 判断 VAR 冲击影响是否衰减。

## 最小例子

VAR(1)：
$$
x_t=A_0+A_1x_{t-1}+e_t
$$
在稳定时可展开为：
$$
x_t=\mu+e_t+A_1e_{t-1}+A_1^2e_{t-2}+\cdots.
$$

## 易混点

- VMA 通常不是新估计的模型，而是稳定 VAR 的等价表示。
- 简约冲击 $e_t$ 和结构冲击 $\varepsilon_t$ 不同。
- 没有稳定性时，无限展开不收敛，IRF 也难以解释为衰减路径。

## 来自课程位置

- [[05_多方程模型Multi-equation Time Series Models#4.4. 脉冲响应函数在VAR中的应用|时间序列 05：VAR 的 VMA 表示]]

## 关联卡片

- [[VAR Model]]
- [[Reduced Form VAR]]
- [[Structural VAR]]
- [[Impulse Response Function]]
- [[Variance Decomposition]]

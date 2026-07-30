---
aliases:
- Impulse Response Function
- IRF
- Impulse Response
- 脉冲响应函数
- 冲击响应函数
tags:
  - concept
  - 时间序列
---
# Impulse Response Function

## 先记一句话

脉冲响应函数就是：**一个冲击发生后，系统中变量在未来各期如何反应的路径**。

## 它是什么

在 VMA 表示中：
$$
x_t=\mu+\sum_{i=0}^{\infty}\Psi(i)\varepsilon_{t-i}.
$$

$\Psi(i)$ 的元素表示第 $i$ 期后某个结构冲击对某个变量的影响。

## 它解决什么判断

IRF 回答：

- 冲击当期影响多大；
- 影响多久衰减；
- 影响方向是否反转；
- 一个变量的冲击如何传导到另一个变量。

## 最小例子

AR(1)：
$$
y_t=a_1y_{t-1}+\varepsilon_t.
$$

$t$ 期一个单位冲击对 $t+j$ 期的影响是：
$$
a_1^j.
$$

## 易混点

- 简约 VAR 的残差不一定是结构冲击，结构 IRF 需要识别约束。
- Cholesky IRF 对变量排序敏感。
- IRF 描述动态响应，不等于因果结论自动成立。

## 来自课程位置

- [[05_多方程模型Multi-equation Time Series Models#1.2. 脉冲响应函数 IRF Impulse Response Analysis|时间序列 05：干预分析中的 IRF]]
- [[05_多方程模型Multi-equation Time Series Models#4.4. 脉冲响应函数在VAR中的应用|时间序列 05：VAR 中的 IRF]]

## 关联卡片

- [[VAR Model]]
- [[VMA]]
- [[Structural VAR]]
- [[Variance Decomposition]]
- [[Intervention Analysis]]

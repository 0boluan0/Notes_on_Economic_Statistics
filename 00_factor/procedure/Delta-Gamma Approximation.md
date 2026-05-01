---
aliases:
- Delta-Gamma Approximation
- Delta-Gamma近似法
- Delta Gamma Approximation
tags:
- procedure
- derivatives
- risk-management
---
# Delta-Gamma Approximation

## 这张卡什么时候用

当组合含期权等 [[Nonlinear Products]]，只用 Delta 近似会漏掉明显二阶风险时，用 Delta-Gamma approximation。

## 输入

- 当前 Delta 向量 $\Delta$。
- Gamma 矩阵 $\Gamma$ 或单因子 Gamma。
- 风险因子变化 $\Delta x$。

## 输出

- 一阶加二阶的 P&L 近似。

## Step 1：先写一阶项

$$
\Delta V_1=\Delta^\top\Delta x
$$

## Step 2：再写二阶项

多因子：

$$
\Delta V_2=\frac{1}{2}\Delta x^\top\Gamma\Delta x
$$

单因子：

$$
\Delta V_2=\frac{1}{2}\Gamma(\Delta S)^2
$$

## Step 3：合并近似

$$
\Delta V\approx \Delta^\top\Delta x+\frac{1}{2}\Delta x^\top\Gamma\Delta x
$$

## 检查点

- Gamma 项不随 $\Delta x$ 正负改变符号，因为包含平方或二次型。
- 这仍是局部泰勒展开，不是完整重新定价。
- 如果波动率也变化，要额外考虑 [[Vega]]；如果时间经过，要考虑 [[Theta]]。

## 来自课程位置

- [[08_操作员如何管理风险暴露]]
- [[14_VaR参数法和模拟法]]

## 关联卡片

- [[Delta]]
- [[Gamma]]
- [[Delta Approximation]]
- [[Cornish-Fisher Expansion]]

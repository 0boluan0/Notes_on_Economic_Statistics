---
aliases:
- Delta Approximation
- Delta近似法
- Delta VaR approximation
tags:
- procedure
- derivatives
- risk-management
---
# Delta Approximation

## 这张卡什么时候用

当产品价值对风险因子的关系近似线性，或只考虑很小的市场变量变化时，用 Delta approximation 估计 P&L。

## 输入

- 当前 Delta 向量 $\Delta$。
- 风险因子变化 $\Delta x$。
- 当前组合价值或头寸规模。

## 输出

- 一阶 P&L 近似 $\Delta V$。

## Step 1：列出风险因子变化

把标的价格、汇率、利率或其他风险因子变化写成向量：

$$
\Delta x=(\Delta x_1,\dots,\Delta x_k)
$$

## Step 2：匹配 Delta 暴露

每个风险因子对应一个一阶敏感度：

$$
\Delta=(\Delta_1,\dots,\Delta_k)
$$

## Step 3：计算线性近似

$$
\Delta V\approx \Delta^\top \Delta x
$$

单因子时就是：

$$
\Delta V\approx \Delta\Delta S
$$

## 检查点

- 标的变化必须和 Delta 的单位一致。
- 如果组合明显非线性或变化幅度大，改用 [[Delta-Gamma Approximation]]。
- Delta approximation 不捕捉波动率变化、时间流逝和利率变化。

## 来自课程位置

- [[08_操作员如何管理风险暴露]]
- [[14_VaR参数法和模拟法]]

## 关联卡片

- [[Delta]]
- [[Linear Products]]
- [[VaR]]

---
aliases:
- Historical Simulation VaR
- 历史模拟法VaR
- 历史模拟法VaR计算
tags:
- procedure
- risk-management
---
# Historical Simulation VaR

## 这张卡什么时候用

题目给出历史收益率、历史损益或历史价格情景，并要求用经验分布取 VaR 分位数时，用这张卡。方法概念见 [[Historical Simulation Method]]。

## 输入

- 当前组合价值或当前持仓。
- $N$ 个历史情景下的收益率、价格变化或组合损益。
- [[Confidence Level|置信水平]] $\alpha$。
- [[Holding Period|持有期]] $h$。
- 是否需要对非线性组合重新定价。

## 输出

- $\operatorname{VaR}_{\alpha,h}$。
- 可选：[[ES]]。
- 可选：用于 [[Backtesting|回测]] 的每日 VaR 序列。

## Step 1：把历史情景变成组合损益

线性组合可以用权重和收益率近似：

$$
\Delta P_t=V_0 w^\top r_t
$$

如果组合含期权或其他非线性产品，不要只做线性加权；应在每个历史情景下重新定价组合。

## Step 2：把损益从差到好排序

设损益从小到大为：

$$
\Delta P_{(1)}\le \Delta P_{(2)}\le \cdots \le \Delta P_{(N)}
$$

损失最严重的情景在左端。

## Step 3：定位尾部分位数

尾部概率为 $1-\alpha$，常用位置：

$$
k=\lceil N(1-\alpha)\rceil
$$

于是：

$$
\operatorname{VaR}_{\alpha}=-\Delta P_{(k)}
$$

若题目要求插值，按相邻 order statistic 插值。

## Step 4：可选计算 ES

取 VaR 尾部更差情景的平均损失：

$$
\operatorname{ES}_{\alpha}\approx -\frac{1}{k}\sum_{i=1}^{k}\Delta P_{(i)}
$$

## 检查点

- VaR 最终报告为正的损失金额。
- $N(1-\alpha)$ 太小时，分位数非常不稳定，要联想到 [[VaR Standard Error]]。
- 历史窗口变化会改变结果，见 [[Observation Window]]。
- 历史模拟不是“无需假设”：它假设历史情景对未来有代表性。

## 常见错误

- 把收益从大到小排，导致取错尾部。
- 对含期权组合仍用线性收益近似。
- 把 99% VaR 取成第 99% 好的收益，而不是 1% 左尾损益。

## 来自课程位置

- [[14_VaR参数法和模拟法]]

## 关联卡片

- [[VaR]]
- [[Historical Simulation Method]]
- [[Weighted Historical Simulation]]
- [[Monte Carlo Simulation VaR]]
- [[EVT VaR Calculation]]

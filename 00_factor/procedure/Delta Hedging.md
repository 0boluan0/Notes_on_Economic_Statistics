---
aliases:
- Delta Hedging
- Delta对冲
- Delta hedge
tags:
- procedure
- derivatives
- risk-management
---
# Delta Hedging

## 这张卡什么时候用

当题目要求构造 Delta 中性组合，或根据期权 Delta 决定买卖多少标的资产时使用。

## 输入

- 期权或组合当前 Delta，记为 $\Delta_P$。
- 每单位对冲工具 Delta，现货通常为 1，期货近似为 1。
- 当前头寸数量。

## 输出

- 需要买入或卖出的对冲工具数量。
- 对冲后净 Delta。

## Step 1：计算组合总 Delta

若持有 $n_i$ 份产品，每份 Delta 为 $\Delta_i$：

$$
\Delta_P=\sum_i n_i\Delta_i
$$

## Step 2：写 Delta 中性条件

若对冲工具每单位 Delta 为 $\Delta_H$，需要数量 $N$：

$$
\Delta_P+N\Delta_H=0
$$

所以：

$$
N=-\frac{\Delta_P}{\Delta_H}
$$

## Step 3：确定买卖方向

$N>0$ 表示买入对冲工具，$N<0$ 表示卖出或做空对冲工具。

## Step 4：设置再平衡规则

Delta 会随标的价格、时间和波动率变化。若 [[Gamma]] 很高，需要更频繁再平衡。

## 常见错误

- 忘记乘以合约数量或合约乘数。
- 把单份期权 Delta 当成组合总 Delta。
- 以为 Delta 中性后 Gamma/Vega 风险也消失。

## 来自课程位置

- [[08_操作员如何管理风险暴露]]

## 关联卡片

- [[Delta]]
- [[Gamma]]
- [[Greeks Hedging Map]]
- [[Delta Approximation]]

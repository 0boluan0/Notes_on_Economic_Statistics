---
aliases:
- Modified Duration Calculation
- 修正久期计算
tags:
- procedure
- fixed-income
type: procedure
---
# Modified Duration Calculation

## 这张卡什么时候用

已经有 [[Macaulay Duration]]，并且要估计利率小幅变化对债券价格的百分比影响时使用。

## 输入

- Macaulay duration $D_M$。
- 年化到期收益率 $y$。
- 每年付息次数 $m$。
- 利率变化 $\Delta y$。

## 输出

- 修正久期 $D_{mod}$。
- 价格百分比变化近似 $\Delta P/P$。

## Step 1：统一周期收益率

$$
y_{period}=\frac{y}{m}
$$

## Step 2：计算修正久期

$$
D_{mod}=\frac{D_M}{1+y/m}
$$

若 $D_M$ 还是周期数，先转成年，或在同一周期口径中保持一致。

## Step 3：估计价格变化

$$
\frac{\Delta P}{P}\approx -D_{mod}\Delta y
$$

利率上升时 $\Delta y>0$，价格变化为负。

## Step 4：需要金额时乘以市值

$$
\Delta P\approx -D_{mod}P\Delta y
$$

若 $\Delta y=0.0001$，得到 1bp 变化金额，即 [[Basis Point Value (BPV)]]。

## 检查点

- 修正久期通常略小于 Macaulay duration。
- 小幅利率变化时，一阶近似更可靠。
- 大幅利率变化要加入 [[Convexity]]。

## 常见错误

- 把 1% 写成 1，而不是 0.01。
- 忽略负号，导致利率上升时价格也上升。
- 把百分比敏感度误当成金额敏感度。

## 来自课程位置

- [[09_利率风险]]

## 关联卡片

- [[Modified Duration]]
- [[Macaulay Duration Calculation]]
- [[Dollar Duration]]
- [[DV01 Hedge Calculation]]

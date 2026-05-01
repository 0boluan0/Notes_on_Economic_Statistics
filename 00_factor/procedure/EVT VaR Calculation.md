---
aliases:
- EVT VaR Calculation
- Extreme Value Theory VaR Calculation
- 极值理论VaR计算
- 极端值理论VaR计算
tags:
- procedure
- risk-management
---
# EVT VaR Calculation

## 这张卡什么时候用

当题目要求估计极高置信度 VaR，且普通历史分位数尾部样本太少时，用 EVT 的 POT/GPD 方法。概念见 [[EVT]] 与 [[GPD]]。

## 输入

- 损失序列 $L_1,\dots,L_T$，损失为正。
- 阈值 $u$。
- 超过阈值的样本数 $n_u$。
- GPD 参数 $\xi$ 与 $\beta$，或可用于估计它们的超额损失样本。
- 置信水平 $\alpha$。

## 输出

- 高置信度 $\operatorname{VaR}_{\alpha}$。
- 可选：[[ES]]。

## Step 1：选阈值

选择足够高但仍有足够超额样本的阈值 $u$。阈值太低，GPD 近似不可信；阈值太高，样本数太少。

## Step 2：提取超额损失

对所有 $L_t>u$ 的观测，定义：

$$
y_i=L_t-u,\qquad i=1,\dots,n_u
$$

## Step 3：估计 GPD 参数

拟合超额损失：

$$
P(Y>y)\approx \left(1+\xi\frac{y}{\beta}\right)^{-1/\xi}
$$

其中 $\xi$ 是形状参数，$\beta$ 是尺度参数。

## Step 4：计算超过阈值的经验概率

$$
\hat p_u=\frac{n_u}{T}
$$

## Step 5：外推 VaR

当目标分位数落在阈值以上时：

$$
\operatorname{VaR}_{\alpha}
=u+\frac{\beta}{\xi}
\left[
\left(\frac{1-\alpha}{\hat p_u}\right)^{-\xi}-1
\right]
$$

若 $\xi$ 接近 0，使用指数尾部极限形式。

## Step 6：可选计算 ES

当 $\xi<1$ 时：

$$
\operatorname{ES}_{\alpha}
=\frac{\operatorname{VaR}_{\alpha}}{1-\xi}
+\frac{\beta-\xi u}{1-\xi}
$$

如果 $\xi\ge 1$，尾部均值不稳定，ES 不应机械报告。

## 检查点

- EVT 是尾部外推，不是普通 VaR 的默认算法。
- 损失方向必须统一；收益左尾要先转成损失右尾。
- $\xi$ 对结果影响很大，必须做阈值敏感性检查。
- 超额样本太少时，结果可能只是精致的猜测。

## 常见错误

- 阈值随手取，没有检查超额样本数量。
- 把 $\alpha$ 和尾部概率 $1-\alpha$ 混用。
- 在 $\xi\ge 1$ 时仍报告稳定 ES。

## 来自课程位置

- [[14_VaR参数法和模拟法]]

## 关联卡片

- [[EVT]]
- [[GPD]]
- [[VaR]]
- [[VaR Method Selection]]
- [[Historical Simulation VaR]]

---
aliases:
- Absolute VaR
- 绝对VaR
- 绝对风险价值
tags:
- concept
- risk-management
---
# Absolute VaR

## 先记一句话

Absolute VaR 直接度量组合未来可能亏掉多少钱，不相对于基准或预期收益做调整。

## 它是什么

在置信水平 $\alpha$ 和持有期 $h$ 下，Absolute VaR 是损失变量 $L$ 的 $\alpha$ 分位数：

$$
\operatorname{VaR}_{\alpha,h}=q_\alpha(L_h)
$$

如果用组合价值变化 $\Delta V$ 表示，常写成：

$$
\operatorname{VaR}_{\alpha,h}=-q_{1-\alpha}(\Delta V_h)
$$

## 解决什么判断

它回答：“在这个 horizon 内，正常市场条件下，损失超过这个金额的概率大约是多少？”

## 最小例子

1 日 99% Absolute VaR = 200 万，意思是按模型估计，未来 1 个交易日损失超过 200 万的概率约为 1%。

## 易混点

- Absolute VaR 关心总损失，不扣除 benchmark 表现；相对 benchmark 的口径见 [[Relative VaR]]。
- VaR 通常报告为正的损失金额，不是收益率的负数本身。
- 只写金额不写 [[Confidence Level]] 和 [[Holding Period]]，这个数字没有完整含义。

## 来自课程位置

- [[12_VAR风险]]

## 关联卡片

- [[VaR]]
- [[Relative VaR]]
- [[VaR Method Selection]]

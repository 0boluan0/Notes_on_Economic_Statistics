---
aliases:
- Kupiec Test
- Kupiec POF Test
- Kupiec Proportion of Failures Test
- Kupiec检验
- 失败比例检验
tags:
- concept
- risk-management
---
# Kupiec Test

## 先记一句话

Kupiec Test 检验 VaR 例外次数是否和名义尾部概率一致。

## 它是什么

在 $n$ 天回测中，若实际损失超过 VaR 的次数为 $x$，名义尾部概率为 $p=1-\alpha$，Kupiec Test 比较观测例外率 $\hat p=x/n$ 和理论例外率 $p$ 是否显著不同。

常用无条件覆盖率统计量：

$$
LR_{uc}
=-2\log
\frac{(1-p)^{n-x}p^x}
{(1-\hat p)^{n-x}\hat p^x}
\sim \chi^2(1)
$$

## 解决什么判断

它回答：“这个 VaR 模型是不是在频率上系统性低估或高估风险？”

## 最小例子

250 天 99% VaR 的理论例外次数大约是 2.5 次。如果实际例外次数远高于这个水平，模型可能低估风险。

## 易混点

- Kupiec Test 只看例外总次数，不看例外是否聚集；聚集问题看 [[Clustering Test]] 或 [[Christoffersen Test]]。
- 例外过少也可能说明模型过度保守，会浪费资本。
- 通过 Kupiec Test 不等于 VaR 模型整体可靠。

## 来自课程位置

- [[12_VAR风险]]

## 关联卡片

- [[Backtesting]]
- [[VaR]]
- [[Clustering Test]]
- [[VaR Standard Error]]

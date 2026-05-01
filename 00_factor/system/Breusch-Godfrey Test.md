---
aliases:
- Breusch-Godfrey Test
- BG Test
- BG检验
- LM自相关检验
tags:
- system
- econometrics
---
# Breusch-Godfrey Test

## 诊断目标

BG 检验用于检验回归残差是否存在指定阶数的自相关，尤其适合 DW 不适用的动态模型。

## 输入

- 原回归残差 $\hat u_t$。
- 要检验的滞后阶数 $p$。
- 原回归中的解释变量。

## 步骤

1. 估计原模型并保存残差。
2. 做辅助回归：把 $\hat u_t$ 对原解释变量和 $\hat u_{t-1},\dots,\hat u_{t-p}$ 回归。
3. 取辅助回归 $R^2$。
4. 计算：

$$
LM=nR^2\sim \chi^2(p)
$$

## 判断

- $p$ 值小：拒绝无自相关，存在指定阶数内的自相关。
- $p$ 值大：没有足够证据说明存在该阶数自相关。

## 易错点

- 辅助回归要保留原解释变量。
- 滞后阶数 $p$ 需要有经济或信息准则依据。
- 发现自相关后仍要判断来源，是误差结构还是模型漏掉动态项。

## 来自课程位置

- [[08_自相关]]

## 关联卡片

- [[Autocorrelation Diagnosis]]
- [[Durbin-Watson Statistic]]
- [[Newey-West]]

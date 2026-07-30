---
aliases:
- KPSS Test
- KPSS
- KPSS检验
- Kwiatkowski-Phillips-Schmidt-Shin Test
tags:
  - concept
  - 时间序列
---
# KPSS Test

## 先记一句话

KPSS 检验就是：**把“平稳”放在原假设的一类平稳性检验**。

## 它是什么

KPSS 与 ADF/DF 的原假设方向相反：

| 检验 | 原假设 |
| --- | --- |
| DF / [[Augmented Dickey-Fuller Test|ADF]] | 存在单位根，非平稳 |
| KPSS | 序列平稳 |

## 它解决什么判断

KPSS 通常用来和 ADF/PP 互相校验：

- ADF 拒绝单位根，KPSS 不拒绝平稳：平稳证据较强。
- ADF 不拒绝单位根，KPSS 拒绝平稳：非平稳证据较强。
- 两者冲突：检查趋势项、结构突变、样本长度和滞后设定。

## 最小直觉

KPSS 把序列拆成趋势部分和误差部分。如果趋势部分像随机游走那样变化，就会拒绝平稳。

## 易混点

- KPSS 的拒绝方向和 ADF 相反。
- “不拒绝平稳”不等于证明平稳，只是没有足够证据反对平稳。
- KPSS 也有水平平稳和趋势平稳版本，要结合图形选择。

## 来自课程位置

- [[06_含趋势的模型#4.3. KPSS检验|时间序列 06：KPSS 检验]]
- [[06_含趋势的模型#4.4. 三检验比较|时间序列 06：DF/ADF 与 KPSS 对比]]

## 关联卡片

- [[Unit Root Test]]
- [[Augmented Dickey-Fuller Test]]
- [[Phillips-Perron Test]]
- [[Stationarity Tests Comparison]]
## 最小例子

把 **KPSS Test** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
## 核心公式

$$\operatorname{KPSS}=\frac{1}{T^2\hat\sigma^2}\sum_{t=1}^{T}S_t^2,\qquad S_t=\sum_{i=1}^{t}\hat u_i.$$

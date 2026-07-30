---
aliases:
- Confidence Ellipsoid
- Confidence Region
- 置信椭球
- 均值向量置信区域
tags:
- concept
- multivariate statistics
---

# Confidence Ellipsoid

>[!note] 一句话记忆
> 置信椭球是多元均值向量的联合置信区域，由 Hotelling $T^2$ 型二次型给出。

## 它是什么

均值向量 $\mu$ 的置信区域常写作
$$
n(\bar X-\mu)'S^{-1}(\bar X-\mu)
\leq
\frac{p(n-1)}{n-p}F_{p,n-p}(\alpha).
$$

这是一组可能的 $\mu$，几何上是椭圆或椭球。

## 解决什么判断

- 多个均值分量的联合不确定性有多大。
- 目标均值 $\mu_0$ 是否落在联合置信区域内。
- 为什么同时置信区间和单变量区间不等价。

## 最小例子

二维均值向量的置信区域是一张椭圆；椭圆方向由 $S$ 的特征向量决定。

## 易混点

- 置信椭球是联合区域，不是各分量区间的简单拼接。
- 单个分量区间包含 0，不一定意味着联合检验不拒绝。

## 来自课程位置

- [[05_ 总体平均向量的推论#1.5. 置信区域与同时置信区间|第5章 3 置信区域与同时置信区间]]

## 关联卡片

- [[Hotelling T2 Test]]
- [[Simultaneous Confidence Intervals]]
- [[Mahalanobis Distance]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[05_ 总体平均向量的推论]]、[[Hotelling T2 Test]]、[[Simultaneous Confidence Intervals]]、[[Mahalanobis Distance]]。

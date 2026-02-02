---
aliases:
- 广义矩估计
- Generalized Method of Moments
tags:
- 计量经济学
- 估计方法
- concept
---
GMM（Generalized Method of Moments，广义矩估计）是一种基于矩条件的参数估计方法，是OLS、2SLS、MLE等估计方法的统一框架。

## 基本思想

通过选择参数θ使得样本矩条件尽可能接近理论矩条件（通常接近0）。

## 矩条件

理论矩条件：$E[m(x_t, \theta)] = 0$

其中m(x_t, θ)是矩条件函数向量。

样本矩条件：$\bar{m}(\theta) = \frac{1}{n} \sum_{t=1}^{n} m(x_t, \theta)$

## GMM估计量

选择θ最小化二次型：

$$J(\theta) = \bar{m}(\theta)' W \bar{m}(\theta)$$

其中W是权重矩阵。

## 权重矩阵W的选择

1. **恒等矩阵**：W = I，得到一致的但非有效的估计
2. **最优权重矩阵**：W = S^{-1}，其中S是矩条件的渐近方差矩阵
   - 最优GMM估计量是渐近有效的
3. **两步估计**：第一步用W = I估计θ，第二步用估计的S计算W

## 最优GMM估计量

$$\hat{\theta}_{GMM} = \arg\min_{\theta} \bar{m}(\theta)' \hat{S}^{-1} \bar{m}(\theta)$$

其中$\hat{S}$是S的一致估计量。

## 与其他方法的关系

1. **OLS**：矩条件E[x_t'(y) = x_t'β] = 0
2. **2SLS/IV**：矩条件E[Z'(y) = X'β] = 0
3. **MLE**：在正态假设下，MLE等价于GMM
4. **GMM**：上述方法的统一框架

## 过度识别检验

当矩条件个数k > 参数个数p时：

检验统计量：$J = n \cdot \bar{m}(\hat{\theta})' \hat{S}^{-1} \bar{m}(\hat{\theta})$

在H₀（模型正确设定）下，J ~ χ²(k-p)

相关链接: [[2SLS]], [[00_factor/concept/Instrumental Variable|工具变量]], [[00_factor/concept/Maximum Likelihood Estimation|极大似然估计]]

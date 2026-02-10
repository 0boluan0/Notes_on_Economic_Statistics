---
aliases:
- Wishart分布
- Wishart 分布
- Wishart Distribution
tags:
- concept
- multivariate statistics
---
# Wishart 分布

## 定义

Wishart 分布是多元情形下的卡方分布（$\chi^2$ 分布）的推广，描述样本协方差矩阵的分布。

如果总体是多元正态分布，那么样本的方差协方差矩阵就服从 Wishart 分布。

### 记号

$(n-1)S \sim W_p(n-1, \Sigma)$

其中：
- $(n-1)S$：样本协方差矩阵的无缩放版本
- $n-1$：自由度（degrees of freedom）
- $\Sigma$：底层正态分布的真实协方差矩阵
- $p$：维度

## 形成原理

如果 $Z_1, \ldots, Z_m$ 是相互独立的 $N_p(0, \Sigma)$ 随机向量：

$\sum_{j=1}^m Z_j Z_j' \sim W_p(m, \Sigma)$

## 性质

### 1. 独立性
在从 $N_p(\mu, \Sigma)$ 抽样时：
- 样本均值 $\bar{X}$ 与样本协方差矩阵 $S$ 相互独立
- $\bar{X} \sim N_p(\mu, \frac{1}{n}\Sigma)$

### 2. 可加性
如果 $A_1 \sim W_{m_1}(\Sigma)$、$A_2 \sim W_{m_2}(\Sigma)$ 且相互独立：

$(A_1 + A_2) \sim W_{m_1 + m_2}(\Sigma)$

这与一维 $\chi^2$ 分布的可加性类似。

### 3. 矩阵变换保持形式
如果 $A \sim W_m(\Sigma)$ 且 $C$ 是可逆矩阵：

$CAC' \sim W_m(C\Sigma C')$

## 与 $\chi^2$ 分布的关系

| 分布 | 形式 | 应用场景 |
|------|------|---------|
| $\chi^2$ | 一元 | 一维样本方差 |
| Wishart | 多元 | 多维样本协方差矩阵 |

- $当 p = 1 时，Wishart 分布退化为 \chi^2 分布$
- Wishart 是多维情形下的"卡方分布矩阵版"

## 应用场景

1. **构造多元检验统计量**：
   - Hotelling $T^2$ 检验
   - Wilks $\Lambda$ 检验
   - 多元方差分析（MANOVA）

2. **样本协方差的推断**：
   - 协方差矩阵的置信区间
   - 协方差相等性检验

3. **贝叶斯统计**：
   - 协方差矩阵的共轭先验

## 相关概念

- [[Multivariate Normal Distribution|多元正态分布]]
- [[Hotelling T2 Test|Hotelling T² 检验]]
- [[Chi-square Distribution|卡方分布]]

## 性质总结

- 如果总体服从多元正态，则 $(n-1)S$ 服从 Wishart
- 自由度越大，分布越集中于真实协方差矩阵
- 样本均值与样本协方差相互独立（正态总体性质）

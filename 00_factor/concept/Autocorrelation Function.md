---
aliases:
- 自相关函数
- ACF
- 自相关函数 ACF
- Autocorrelation Function
tags:
- 时间序列
- 统计学
- concept
---
自相关函数（Autocorrelation Function, ACF）是衡量时间序列自身在不同滞后时间上相关程度的函数。

## 定义

$$\rho_k = \frac{\text{Cov}(y_t, y_{t+k})}{\sqrt{\text{Var}(y_t)\text{Var}(y_{t+k})}}$$

对于平稳时间序列，ACF只依赖于滞后阶数k。

## 性质

1. **ρ₀ = 1**：时间序列与自身的相关系数为1
2. **对称性**：ρ_k = ρ_{-k}
3. **范围**：-1 ≤ ρ_k ≤ 1
4. **渐近独立**：对于白噪声，ρ_k在k > 0时趋近于0

## 在ARMA模型中的应用

- **AR(p)过程**：ACF表现为拖尾（逐渐衰减）
- **MA(q)过程**：ACF在q阶后截尾（为0）
- **ARMA(p,q)过程**：ACF表现为拖尾

## 样本ACF估计

$$\hat{\rho}_k = \frac{\sum_{t=k+1}^{T} (y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^{T} (y_t - \bar{y})^2}$$

## 应用

1. 检验序列相关性
2. 识别ARMA模型阶数
3. 评估模型拟合效果
4. [[白噪声检验]]

相关链接: [[偏自相关函数]], [[ARMA]], [[白噪声检验]]

## source_notes

- [[03_平稳时间序列模型#0.回忆用]]（ACF 计算提示）
- [[03_平稳时间序列模型#1. 引言 Introduction]]

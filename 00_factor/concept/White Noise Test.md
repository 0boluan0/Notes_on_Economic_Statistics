---
aliases:
- 白噪声检验
- White Noise Test
- Portmanteau Test
tags:
- concept
- time-series
- statistical-test
---
# White Noise Test

## 一句话记忆

白噪声检验检验序列在给定若干滞后内是否没有显著自相关；它通常是模型残差诊断，而不是“证明序列完全随机”。

## 白噪声过程定义

{ε_t}是白噪声过程当且仅当：

1. **零均值**：E(ε_t) = 0，对所有t
2. **常数方差**：Var(ε_t) = σ²，对所有t
3. **无自相关**：Cov(ε_t, ε_{t-k}) = 0，对所有k ≠ 0

## 常见统计量

### 1. Ljung-Box检验

**统计量：**

$Q = T(T+2) \sum_{k=1}^{m} \frac{\hat{\rho}_k^2}{T-k}$

其中：
- T：样本量
- m：检验的滞后阶数
- $\hat{\rho}_k$：样本自相关系数

在无参数估计、样本足够大等近似条件下，$Q$ 近似服从 $\chi^2(m)$；若检验的是含参数模型的残差，自由度通常需要扣除已估计参数数目。

**原假设H₀**：序列是白噪声（所有自相关系数为0）

### 2. McLeod–Li 检验

用于检验平方序列的自相关，检测ARCH效应：

$Q = T(T+2) \sum_{k=1}^{m} \frac{r_k^2}{T-k}$

其中$r_k$是平方序列的自相关系数。

### 3. ARCH-LM 检验

专门检验条件异方差时，可使用 [[ARCH LM Test]]；它与检验线性自相关的 Ljung–Box 并非同一个原假设。

## 结论如何读

- 拒绝 $H_0$：给定滞后范围内仍有线性相关结构，或模型遗漏了动态项。
- 不拒绝 $H_0$：没有足够证据拒绝“这些自相关为零”，不等于证明独立同分布。

常见用途是检查 [[ARMA]] 残差的均值动态，以及检查残差平方中是否仍有 [[Volatility Clustering]]。

## 注意事项

1. **滞后阶数选择**：$m$ 过小可能漏掉较远相关，过大则降低检验功效。
2. **小样本与参数估计**：卡方近似可能较差，残差检验还要考虑模型参数数量。
3. **多重检验**：同时查看多个滞后阶数时，不应只挑一个显著结果叙述。

相关链接: [[Autocorrelation Function|自相关函数]], [[ARCH]], [[GARCH]], [[Ljung-Box Test|Ljung-Box检验]], [[Partial Autocorrelation Function|偏自相关函数]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```

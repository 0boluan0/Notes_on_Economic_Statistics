---
aliases:
- 随机效应模型
- 随机效应
- Random Effects Model
- Random
tags:
- 计量经济学
- 面板数据
- concept
---
随机效应（Random Effects, RE）模型是面板数据模型的一种，假设个体效应是随机变量，与解释变量不相关。

## 模型形式

$$y_{it} = \beta_0 + x_{it}'\beta + \alpha_i + \varepsilon_{it}$$

其中：
- y_{it}：个体i在时期t的观测值
- β_0：常数项
- x_{it}：k×1维解释变量向量
- β：k×1维系数向量
- α_i ~ N(0, σ_α²)：个体i的随机效应
- ε_{it} ~ N(0, σ_ε²)：误差项

## 关键假设

1. **α_i与x_{it}不相关**：随机效应是外生的
2. **样本代表性**：样本对总体具有代表性
3. **误差项假设**：α_i和ε_{it}独立

## 合并误差项

定义复合误差项：v_{it} = α_i + ε_{it}

$$\text{Var}(v_{it}) = \text{Var}(\alpha_i) + \text{Var}(\varepsilon_{it}) = \sigma_{\alpha}^2 + \sigma_{\varepsilon}^2$$

个体内自相关：

$$\text{Cov}(v_{it}, v_{is}) = \sigma_{\alpha}^2, \quad t \neq s$$

## 估计方法：GLS

由于误差项存在异方差和自相关，使用广义最小二乘（GLS）：

$$\hat{\beta}_{RE} = (X'\Omega^{-1} X)^{-1} X'\Omega^{-1} y$$

其中Ω是v的协方差矩阵。

## FGLS（可行GLS）

由于σ_α²和σ_ε²未知，需要先估计：

1. 先用OLS或FE估计，得到残差
2. 估计σ_α²和σ_ε²
3. 用估计的Ω进行GLS

## σ_α²和σ_ε²的估计

### 估计σ_ε²

从组内回归的残差估计。

### 估计σ_α²

$$\hat{\sigma}_{\alpha}^2 = \frac{1}{n-k-1} \sum_{i=1}^{N} (T_i \cdot \bar{u}_i^2) - \hat{\sigma}_{\varepsilon}^2$$

## 性质

1. **效率性**：在假设正确时，RE估计量更有效（方差更小）
2. **一致性**：在假设正确时，RE估计量是一致的
3. **局限性**：如果α_i与x相关，RE估计量有偏且不一致

## 豪斯曼检验

用于选择固定效应还是随机效应模型。

**H₀**：α_i与x_{itkt}不相关（应使用RE）
**H₁**：α_i与x_{itkt}相关（应使用FE）

## 适用场景

1. **个体效应与解释变量不相关**：如个体异质性为随机
2. **样本代表总体**：样本对总体具有代表性
3. **估计不随时间变化的变量**：可以估计性别、地区等

## 与固定效应比较

| 特性 | 随机效应 | 固定效应 |
|------|----------|----------|
| 假设 | α_i与x不相关 | α_i与x相关 |
| 估计方法 | GLS | 组内变换/OLS |
| 有效性 | 假设正确时更有效 | 总是一致 |
| 适用 | α_i与x不相关 | α_i与x相关 |

相关链接: [[00_factor/concept/Fixed Effects Model|固定效应]], [[00_factor/concept/Hausman Test|豪斯曼检验]], [[00_factor/concept/Panel Data Model|面板数据模型]]

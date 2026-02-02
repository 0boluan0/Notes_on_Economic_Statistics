---
aliases:
- Box-Jenkins建模方法
- Box-Jenkins方法
- Box
- Box-Jenkins Method
tags:
- procedure
- 06_时间序列分析
---
# Box-Jenkins方法

## 适用场景

建立和估计ARMA/GARCH模型的标准化三阶段流程，适用于平稳时间序列的建模、预测和诊断。

## 所需数据/条件

- 平稳时间序列 ${y_t}_{t=1}^T$
- 或已中心化的序列 ${\epsilon_t}_{t=1}^T$（均值为零）

## 计算步骤

### 阶段一：识别（Identification）

#### 步骤 1：绘制时序图

可视化数据趋势和基本特征：
- 检查是否存在趋势（需差分或去趋势）
- 检查异常值和结构断点
- 观察波动聚集性

**注意点**：非平稳序列需先差分或去趋势。

#### 步骤 2：检验平稳性

使用ADF检验或PP检验：
- 原假设 $H_0$：存在单位根（非平稳）
- 备择假设 $H_1$：无单位根（平稳）

若拒绝原假设，序列平稳；若未拒绝，继续检验差分序列。

**注意点**：平稳性是ARMA建模的前提。

#### 步骤 3：计算和绘制ACF、PACF

计算样本自相关函数和偏自相关函数，绘制到滞后 $K$：

**ACF**：
$$
\hat{\rho}_k = \frac{\sum_{t=k+1}^{T} (y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^{T} (y_t - \bar{y})^2}
$$

**PACF**（通过回归估计）：
估计 $y_t = \phi_{k1} y_{t-1} + \cdots + \phi_{kk} y_{t-k} + e_t$，得到 $\hat{\phi}_{kk}$

**注意点**：一般取 $K = 20$ 或 $K = \sqrt{T}$。

#### 步骤 4：初步识别模型

根据ACF和PACF模式：

| ACF模式 | PACF模式 | 可能模型 |
|---------|---------|---------|
| q阶后截尾为0 | 拖尾 | MA(q) |
| 拖尾（指数衰减） | p阶后截尾为0 | AR(p) |
| 拖尾 | 拖尾 | ARMA(p,q) |

**注意点**：纯AR或纯MA只是特例，实际常为ARMA。

### 阶段二：估计（Estimation）

#### 步骤 5：估计模型参数

对候选模型进行参数估计：
- **纯AR(p)**：使用OLS估计
- **含MA部分**：使用MLE（极大似然估计）

估计得到参数 $\hat{\phi}_1,\dots,\hat{\phi}_p$ 和 $\hat{\theta}_1,\dots,\hat{\theta}_q$。

**注意点**：需检验可逆性（MA根在单位圆外）。

#### 步骤 6：检验参数显著性

使用t检验检验各参数：
$$
t_{\phi_i} = \frac{\hat{\phi}_i}{se(\hat{\phi}_i)}
$$

若 $|t_{\phi_i}| > t_{\alpha/2}$，参数显著。

**注意点**：不显著参数可考虑剔除以简化模型。

#### 步骤 7：使用信息准则比较模型

计算各候选模型的AIC或BIC：

$$
\text{AIC} = T \ln(\hat{\sigma}^2) + 2k
$$
$$
\text{BIC} = T \ln(\hat{\sigma}^2) + k \ln T
$$

其中 $k$ 为参数个数。

**注意点**：选择AIC或BIC最小的模型。

### 阶段三：诊断检验（Diagnostic Checking）

#### 步骤 8：残差分析

计算模型残差：
$$\hat{\epsilon}_t = y_t - \hat{y}_t, \quad t=1,\dots,T
$$

检验残差性质：
- **零均值检验**：$\bar{\epsilon} \approx 0$
- **正态性检验**：使用Jarque-Bera检验或QQ图
- **白噪声检验**：使用Ljung-Box或Box-Pierce检验

**注意点**：残差应为白噪声才能说明模型充分。

#### 步骤 9：Ljung-Box检验

检验残差是否为白噪声（无自相关）：

$$
Q = T \sum_{k=1}^{s} \hat{\rho}_{\epsilon,k}^2 \xrightarrow{d} \chi^2(s)
$$

其中 $s$ 为检验滞后期数（如10或20）。

**注意点**：若 $Q > \chi^2_{\alpha}(s)$，拒绝白噪声假设。

#### 步骤 10：预测（可选）

一步预测：
$$\hat{y}_{T+1|T} = \hat{\phi}_1 y_T + \cdots + \hat{\phi}_p y_{T-p+1} + \hat{\theta}_1 \hat{\epsilon}_T + \cdots
$$

多步预测需递推公式。

**注意点**：多步预测收敛到序列均值。

## 关键公式

**ARMA(p,q)模型**：
$$
y_t = \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \cdots + \theta_q \epsilon_{t-q}
$$

**Yule-Walker方程**：
$$
\gamma_k = \phi_1 \gamma_{k-1} + \cdots + \phi_p \gamma_{k-p} + \rho_k \sigma^2, \quad k \ge 1
$$

**Ljung-Box Q统计量**：
$$
Q = T(T+2) \sum_{k=1}^{s} \frac{\hat{\rho}_k^2}{T - k} \xrightarrow{d} \chi^2(s)
$$

**信息准则**：
$$
\text{AIC} = T \ln(\hat{\sigma}^2) + 2(p+q)
$$
$$
\text{SBC} = T \ln(\hat{\sigma}^2) + (p+q) \ln T
$$

## 常见问题

1. **平稳性忽略**：对非平稳序列建模导致谬误结果。
2. **过拟合**：增加阶数总降低残差，但预测能力可能下降。
3. **信息准则选择**：AIC倾向于选择较复杂模型，BIC更保守。
4. **残差非白噪声**：模型设定错误或遗漏变量。
5. **结构断点**：样本期间模型参数可能变化，需分段建模。

## 相关概念
[[Stationary Time Series|平稳时间序列]]
[[00_factor/procedure/ARMA Model Identification Steps|ARMA模型识别步骤]]
[[00_factor/procedure/GARCH Model Estimation Steps|GARCH模型估计步骤]]
[[00_factor/concept/Unit Root Test|单位根]]

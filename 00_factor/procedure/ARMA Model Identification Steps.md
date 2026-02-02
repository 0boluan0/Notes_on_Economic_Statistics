---
aliases:
- ARMA模型定阶步骤
- ARMA模型识别步骤
- ARMA
- ARMA模型识别
- ARMA Model Identification Steps
tags:
- procedure
- 06_时间序列分析
- 经济
---
# ARMA模型识别步骤

## 适用场景

识别ARMA(p,q)模型的自回归阶数 $p$ 和移动平均阶数 $q$，用于平稳时间序列建模。

## 所需数据/条件

- 平稳时间序列 ${y_t}_{t=1}^T$
- 样本自相关函数（ACF）和偏自相关函数（PACF）

## 计算步骤

### 步骤 1：检验序列平稳性

使用ADF检验或PP检验确认序列平稳。

**不平稳时**：
- 进行差分：$\Delta y_t = y_t - y_{t-1}$
- 对差分序列重复检验，直到平稳

**注意点**：ARMA建模要求序列平稳，非平稳序列需先差分至平稳。

### 步骤 2：绘制ACF和PACF图

计算样本ACF和PACF，并绘制到滞后 $k=0,1,\dots,K$：

**ACF公式**：
$$ \hat{\rho}_k = \frac{\sum_{t=k+1}^T (y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^T (y_t - \bar{y})^2} $$

**PACF计算**：
通过估计辅助回归 $y_t = \phi_{k1}y_{t-1} + \cdots + \phi_{kk}y_{t-k} + e_t$ 得到 $\hat{\phi}_{kk}$

**注意点**：$K$ 通常取样本量的10%-20%，如 $K = 20$。

### 步骤 3：观察ACF模式

根据ACF图形特征初步判断：

| ACF模式 | 可能模型 | 说明 |
|---------|---------|------|
| 截尾（$q$阶后为0） | MA(q) | ACF在 $q$ 处突然降为0 |
| 拖尾（指数衰减） | AR(p) 或 ARMA(p,q) | ACF缓慢衰减至0 |
| 阻尼阻尼衰减 | AR(p) | 正负交替衰减 |

**注意点**：纯AR(p)的ACF拖尾，纯MA(q)的ACF截尾。

### 步骤 4：观察PACF模式

根据PACF图形特征初步判断：

| PACF模式 | 可能模型 | 说明 |
|---------|---------|------|
| 截尾（$p$阶后为0） | AR(p) | PACF在 $p$ 处突然降为0 |
| 拖尾 | MA(q) 或 ARMA(p,q) | PACF缓慢衰减至0 |

**注意点**：AR(p)模型的PACF在 $p$ 阶处截尾是关键识别特征。

### 步骤 5：初步定阶

根据ACF和PACF模式组合确定 $(p,q)$：

**纯AR(p)**：
- ACF拖尾（缓慢衰减）
- PACF截尾（在 $p$ 阶后为0）

**纯MA(q)**：
- ACF截尾（在 $q$ 阶后为0）
- PACF拖尾（缓慢衰减）

**ARMA(p,q)**：
- ACF和PACF都拖尾
- 需要尝试不同组合

**注意点**：实际数据常为ARMA(p,q)而非纯AR或纯MA。

### 步骤 6：估计候选模型

对候选 $(p_1,q_1), (p_2,q_2), \dots$ 模型进行参数估计：
- 纯AR部分：使用OLS
- 含MA部分：使用MLE或数值方法

**注意点**：MA部分需用MLE，OLS无法直接估计。

### 步骤 7：使用信息准则比较模型

计算各候选模型的AIC或SBC：

$$ \text{AIC} = T \ln(\hat{\sigma}^2) + 2(p+q) $$
$$ \text{SBC} = T \ln(\hat{\sigma}^2) + (p+q) \ln T $$

其中 $\hat{\sigma}^2$ 为残差方差估计。

**选择准则**：选择AIC或SBC最小的模型。

**注意点**：SBC通常选择更简单的模型（惩罚更重）。

### 步骤 8：诊断检验

对选定模型进行残差诊断：

1. **残差ACF检验**：使用Ljung-Box检验检验残差是否为白噪声
2. **残差正态性**：使用Jarque-Bera检验
3. **残差独立性**：检查Durbin-Watson统计量

**注意点**：若残差不满足白噪声假设，需重新定阶。

## 关键公式

**样本ACF**：
$$ \hat{\rho}_k = \frac{\sum_{t=k+1}^T (y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^T (y_t - \bar{y})^2} $$

**样本PACF（回归法）**：
通过 $y_t = \phi_{k1}y_{t-1} + \cdots + \phi_{kk}y_{t-k} + e_t$ 估计 $\hat{\phi}_{kk}$

**AIC信息准则**：
$$ \text{AIC} = T \ln(\sum \hat{\epsilon}_t^2 / T) + 2 \times \text{参数个数} $$

**Ljung-Box Q统计量**：
$$ Q = T(T+2) \sum_{k=1}^s \frac{\hat{\rho}_k^2}{T-k} $$

## 常见问题

1. **阶数不确定**：ACF拖尾和截尾可能不明显，需尝试多个组合。
2. **过度拟合**：增加p和q总降低残差，但导致过拟合。
3. **小样本偏差**：小样本下ACF和PACF估计不稳定。
4. **非平稳序列**：非平稳序列的ACF和PACF行为异常。

## 相关概念
[[自相关函数]]
[[偏自相关函数]]
[[Stationary Time Series|平稳时间序列]]
[[GARCH模型估计步骤]]
[[Box-Jenkins方法]]

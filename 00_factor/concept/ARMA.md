---
aliases:
- 自回归移动平均模型
- Autoregressive Moving Average Model
- ARMA模型
- Autoregressive Moving Average (ARMA) Model
- ARMA
- 自回归滑动平均模型
tags:
- 时间序列
- 波动建模
- concept
---
ARMA(p,q)模型是自回归移动平均模型，结合了AR(p)和MA(q)两种模型的特点，用于描述平稳时间序列。

## 模型形式

$$y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \varepsilon_t$$

其中：
- y_t：时间序列在t期的值
- c：常数项
- φ_i：自回归系数
- θ_j：移动平均系数
- ε_t：白噪声误差项

## 平稳性条件

ARMA模型平稳性仅由AR部分决定，要求特征方程的根在单位圆外：

$$1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p = 0$$

所有根|z| > 1。

## 可逆性条件

可逆性仅由MA部分决定，要求可逆性方程的根在单位圆外。

## ACF和PACF特征

- **ACF**：拖尾（逐渐衰减）
- **PACF**：拖尾（逐渐衰减）

## 参数估计

常用方法：
1. **最大似然估计（MLE）**
2. **条件最小二乘法**
3. **Yule-Walker方程**

## 模型选择

使用信息准则：
- **AIC**：-2ln(L) + 2k
- **BIC**：-2ln(L) + k·ln(T)

相关链接: [[00_factor/concept/Autocorrelation Function|自相关函数]], [[00_factor/concept/Partial Autocorrelation Function|偏自相关函数]], [[ARCH]], [[GARCH]], [[00_factor/concept/Stationarity|平稳性]], [[AIC]], [[BIC]]

## source_notes

- [[03_平稳时间序列模型#1.1 自回归移动平均模型ARMA(p,q) model]]

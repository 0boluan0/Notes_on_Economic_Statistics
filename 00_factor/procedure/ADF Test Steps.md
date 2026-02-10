---
aliases:
- ADF检验步骤
- Augmented Dickey-Fuller检验
- ADF
- ADF Test Steps
tags:
- procedure
- 01_Econometrics
- 06_时间序列分析
---
# ADF检验步骤

## 适用场景

检验时间序列是否存在单位根，判断序列是否平稳。是建立时间序列模型（ARMA、GARCH）的必要前提。

## 所需数据/条件

- $时间序列 \{y_t\}_{t=1}^T$
- 显著性水平 $\alpha$（通常5%）
- 滞后阶数 $p$（需预先确定）

## 计算步骤

### 步骤 1：确定检验模型形式

根据数据特征选择模型形式：

**模型1（无常数项、无趋势）**：
$ \Delta y_t = \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t $

**模型2（有常数项、无趋势）**：
$ \Delta y_t = \alpha + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t $

**模型3（有常数项和趋势）**：
$ \Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t $

**注意点**：$原假设 H_0: \gamma = 0（存在单位根），备择 H_1: \gamma < 0（序列平稳）。$

### 步骤 2：确定滞后阶数 $p$

使用信息准则选择最优滞后阶数：

$ \text{AIC}(p) = T \ln\left(\frac{\text{RSS}_p}{T}\right) + 2p $
$ \text{SBC}(p) = T \ln\left(\frac{\text{RSS}_p}{T}\right) + p \ln T $

选择使信息准则最小的 $p$ 值。

**注意点**：一般从最大可能阶数开始，逐步减少，找到最优值。

### 步骤 3：估计ADF回归模型

使用OLS估计选定的ADF模型，得到：
- $\hat{\gamma}$（关键系数）
- 标准误 $se(\hat{\gamma})$
- 残差平方和 $\text{RSS}$

**注意点**：$\gamma = 0 等价于原序列存在单位根。$

### 步骤 4：构造t统计量

$ \tau = \frac{\hat{\gamma}}{se(\hat{\gamma})} $

**注意点**：此统计量不服从标准t分布，而是服从Dickey-Fuller $\tau$ 分布。

### 步骤 5：获取临界值

根据样本量 $T$、显著性水平 $\alpha$ 和模型形式查ADF临界值表：

**临界值示例**（近似）：
当 $T=500, \alpha=0.05$ 时：
- 模型1：$c_1 \approx -1.95$
- 模型2：$c_2 \approx -2.86$
- 模型3：$c_3 \approx -3.41$

**注意点**：临界值为负值，拒绝域在左尾。

### 步骤 6：进行假设检验

**检验规则**：
- 若 $\tau < c_{\alpha}$（统计量小于临界值），拒绝 $H_0$
- 结论：序列平稳（无单位根）
- 若 $\tau \ge c_{\alpha}$，无法拒绝 $H_0$
- 结论：序列非平稳（存在单位根）

**注意点**：ADF检验是左单侧检验，因为 $\gamma < 0$ 时序列平稳。

### 步骤 7：处理多阶单位根（如需）

若未能拒绝原假设，可能存在更高阶单位根 I(d), d > 1：

1. 对序列做一阶差分：$\Delta y_t = y_t - y_{t-1}$
2. 对差分序列重复ADF检验
3. 若差分后平稳，则原序列为 I(1)

**注意点**：单位根检验应从最低阶开始，逐步检验。

## 关键公式

**ADF模型（模型2）**：
$\Delta y_t = \alpha + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t$

**检验统计量**：
$ \tau = \frac{\hat{\gamma}}{se(\hat{\gamma})} $

**PP检验统计量（替代方法）**：
$ Z_\gamma = \frac{\tau \cdot \sigma^2 - T \cdot \hat{\gamma}^2}{\sigma \sqrt{T}} $

## 常见问题

1. **模型形式选择**：错误选择模型形式（是否含趋势）导致检验结论偏差。
2. **滞后阶数**：p 太小导致误差项自相关，p 太大损失自由度。
3. **结构断点**：存在结构变化时，ADF检验可能失效。
4. **低功效**：小样本下，ADF检验容易接受原假设（第二类错误）。

## 相关概念
[[Unit Root Test|单位根]]
[[Cointegration|协整]]
[[Phillips-Perron Test|PP检验]]
[[Unit Root Test|平稳性检验]]

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

---
aliases:
- GARCH模型估计步骤
- GARCH
- GARCH模型估计
tags:
- procedure
- 06_时间序列分析
- 经济
---
# GARCH模型估计步骤

## 适用场景

建模金融时间序列的条件异方差（波动率聚集）特征，适用于存在波动聚集性的收益序列。

## 所需数据/条件

- 均值为零（或中心化）的收益序列 ${\epsilon_t}_{t=1}^T$
- 波动率模型设定（通常为GARCH(1,1)）

## 计算步骤

### 步骤 1：中心化收益序列

若序列均值不为零，减去样本均值：
$$ \epsilon_t = y_t - \bar{y}, \quad \bar{y} = \frac{1}{T}\sum_{t=1}^T y_t $$

**注意点**：GARCH建模假设均值为零或已中心化。

### 步骤 2：设定GARCH(1,1)模型

最常用的GARCH(1,1)模型：

$$ \epsilon_t = \sigma_t u_t, \quad u_t \sim N(0,1) $$
$$ \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2 $$

约束条件：$\omega > 0, \alpha \ge 0, \beta \ge 0, \alpha + \beta < 1$

**注意点**：$\alpha$ 衡量过去的冲击影响，$\beta$ 衡量过去的波动率持续性。

### 步骤 3：初始化参数

设置参数初始值：
- $\omega_0 = \hat{\sigma}_{\text{OLS}}^2 \times (1 - \alpha_0 - \beta_0)$
- $\alpha_0 = 0.1$（通常较小）
- $\beta_0 = 0.85$（通常较大，接近1）

**注意点**：初始值需满足约束条件。

### 步骤 4：写对数似然函数

对数似然函数为：
$$ \ell(\omega,\alpha,\beta) = -\frac{1}{2}\sum_{t=1}^T \left[ \ln(2\pi) + \ln(\sigma_t^2) + \frac{\epsilon_t^2}{\sigma_t^2} \right] $$

其中 $\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$，设定 $\sigma_0^2 = \frac{1}{T}\sum_{t=1}^T \epsilon_t^2$。

**注意点**：对数似然计算是递归的，需从 $t=1$ 开始依次计算。

### 步骤 5：构造约束优化问题

极大化对数似然，满足约束：
$$ \max_{\omega,\alpha,\beta} \ell(\omega,\alpha,\beta) $$
$$ \text{s.t. } \omega > 0, \alpha \ge 0, \beta \ge 0, \alpha + \beta < 1 $$

**注意点**：常用方法是对参数进行变换确保无约束，如 $\omega = \exp(a), \alpha = \exp(b)/(1+\exp(b)+\exp(c))$。

### 步骤 6：使用数值优化算法求解

**常用算法**：
- BFGS（Broyden-Fletcher-Goldfarb-Shanno）
- L-BFGS（带线性约束）
- 梯度下降法
- 牛顿法

**算法流程**：
1. 从初始值开始
2. 计算梯度 $\nabla \ell$
3. 更新参数
4. 检查收敛性
5. 重复2-4直到收敛

**注意点**：对数似然可能非凸，需小心选择初始值。

### 步骤 7：验证参数约束

检查估计的参数：
- $\hat{\omega} > 0$
- $\hat{\alpha} \ge 0$
- $\hat{\beta} \ge 0$
- $\hat{\alpha} + \hat{\beta} < 1$

**违反约束**：
- 使用约束优化重新估计
- 或使用参数变换

**注意点**：若 $\hat{\alpha} + \hat{\beta} \ge 1$，模型非平稳。

### 步骤 8：计算拟合统计量

**标准化残差**：
$$ z_t = \frac{\epsilon_t}{\hat{\sigma}_t} $$

检验 $z_t$ 是否为标准正态：
- 均值是否为0
- 方差是否为1
- Ljung-Box检验是否无自相关

**注意点**：残差正态性假设对置信区间构建很重要。

### 步骤 9：预测未来波动率

给定当前信息，预测未来波动率：

**一步预测**：
$$ \hat{\sigma}_{T+1}^2 = \hat{\omega} + \hat{\alpha} \epsilon^2_T + \hat{\beta} \hat{\sigma}_T^2 $$

**h步预测**（假设 $\epsilon_{T+j} = 0$）：
$$ \hat{\sigma}_{T+j}^2 = \frac{\hat{\omega}}{1 - \hat{\alpha} - \hat{\beta}} + (\hat{\sigma}_T^2 - \frac{\hat{\omega}}{1 - \hat{\alpha} - \hat{\beta}})(\hat{\alpha} + \hat{\beta})^{j-1} $$

**长期平均波动率**：
$$ \sigma_{\infty}^2 = \frac{\hat{\omega}}{1 - \hat{\alpha} - \hat{\beta}} $$

**注意点**：长期均值是波动率回归的目标水平。

## 关键公式

**GARCH(1,1)条件方差**：
$$ \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2 $$

**对数似然函数**：
$$ \ell(\theta) = -\frac{1}{2}\sum_{t=1}^T \left[ \ln(2\pi) + \ln(\sigma_t^2) + \frac{\epsilon_t^2}{\sigma_t^2} \right] $$

**无条件方差（长期均值）**：
$$ \sigma^2_{\infty} = \frac{\omega}{1 - \alpha - \beta} $$

**波动率半衰期**：
$$ \text{Half-life} = \frac{\ln(0.5)}{\ln(\alpha + \beta)} $$

## 常见问题

1. **初始值敏感**：不同初始值可能收敛到不同局部最优。
2. **数值不稳定**：GARCH对数似然在某些参数区域很平坦。
3. **约束违反**：估计参数可能不满足平稳性条件。
4. **残差非正态**：标准化残差常表现出厚尾，需用t分布。

## 相关概念
[[条件异方差]]
[[波动率聚集]]
[[EWMA波动率估计]]
[[ARCH]]

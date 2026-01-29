---
aliases:
  - Autoregressive Distributed Lag Model
  - 自回归分布滞后模型
tags:
  - 计量经济学
  - 时间序列
---

ADL（Autoregressive Distributed Lag，自回归分布滞后）模型是包含被解释变量滞后项和解释变量滞后项的动态回归模型。

## 模型形式

$$y_t = \alpha + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{j=0}^{q} \beta_j x_{t-j} + \varepsilon_t$$

其中：
- y_t：被解释变量
- x_t：解释变量
- p：y_t的滞后阶数
- q：x_t的滞后阶数（包括当期）
- ε_t：误差项

## 特例

### 1. 适应性预期模型

假设适应性预期：$x_t^* = \lambda x_{t-1} + (1-\lambda)x_t$

ADL(1,0)形式：
$$y_t = \gamma \delta \lambda y_{t-1} + \gamma (1-\lambda)x_t - \gamma \delta \lambda x_{t-1} + \varepsilon_t$$

### 2. 部分调整模型

假设部分调整：$y_t - y_{t-1} = \delta(y_t^* - y_{t-1})$

ADL(1,1)形式：
$$y_t = (1-\delta)y_{t-1} + \delta \alpha_0 + \delta \alpha_1 x_t + \varepsilon_t$$

## 模型性质

1. **长期乘数**：$\theta = \sum_{j=0}^{q} \beta_j / (1 - \sum_{i=1}^{p} \phi_i)$
2. **短期乘数**：β_0（当期影响）
3. **调整速度**：由φ_i和β_j共同决定

## 估计问题

如果误差项存在自相关，OLS估计有偏且不一致。

### Durbin h检验

检验ADL模型中的自相关问题：

$$h = (1 - \frac{d}{2}) \sqrt{\frac{T}{1 - T \hat{\phi}^2}}$$

其中d是Durbin-Watson统计量，$\hat{\phi}$是y_{t-1}的估计系数。

### 工具变量估计法

使用y_{t-2}、x_{t-1}等作为工具变量。

## 应用

1. **动态乘数分析**：分析政策冲击的传导路径
2. **短期和长期效应**：区分即时影响和累积影响
3. **政策评估**：评估财政政策、货币政策的动态效应

相关链接: [[分布滞后模型]], [[工具变量]], [[自相关]]

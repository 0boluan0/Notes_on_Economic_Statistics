---
aliases:
- Error Correction Model
- ECM
- 误差修正模型
- 误差纠正机制(ECM)
tags:
- 计量经济学
- 时间序列
- 协整
- concept
---
误差纠正机制（Error Correction Model，ECM）是描述协整时间序列之间长期均衡关系和短期动态调整过程的模型，将水平关系和差分关系结合在一起。

## 基本思想

### 协整关系

如果两个I(1)过程$y_t$和$x_t$协整：
- 存在长期均衡关系
- 残差平稳：$\epsilon_t = y_t - \beta x_t \sim I(0)$

### 误差纠正

短期偏离均衡时：
- 系统有力量纠正偏离
- 向长期均衡回归。~~回复~~

## 单变量ECM

### 基本形式

对于两个协整变量$y_t$和$x_t$：

$\Delta y_t = \alpha + \beta \Delta x_t + \gamma (y_{t-1} - \hat{\beta} x_{t-1}) + u_t$

其中：
- $\Delta y_t = y_t - y_{t-1}$：$y$的一阶差分
- $\Delta x_t = x_t - x_{t-1}$：$x$的一阶差分
- $y_{t-1} - \hat{\beta} x_{t-1}$：协整误差（滞后一期）
- $\gamma$：误差纠正系数
- $u_t$：白噪声

### 协整误差项

$\epsilon_{t-1} = y_{t-1} - \hat{\beta} x_{t-1}$

表示上一期偏离长期均衡的程度。

### 误差纠正系数

$\gamma$的含义：
- $\gamma < 0$：误差纠正机制，正向偏离会减少$y_t$
- $\gamma = 0$：无误差纠正
- $\gamma > 0$：误差扩大（不稳定）

通常期望$\gamma < 0$。

## 从协整关系推导ECM

### 第一步：估计协整关系

$y_t = \beta x_t + \epsilon_t$

得到$\hat{\beta}$和残差$\hat{\epsilon}_t$。

### 第二步：构建ECM

$\Delta y_t = \alpha + \beta \Delta x_t + \gamma \hat{\epsilon}_{t-1} + u_t$

### 解释

1. **短期效应**：$\Delta x_t$的系数$\beta$
2. **长期均衡**：协整误差项$\hat{\epsilon}_{t-1}$
3. **调整速度**：误差纠正系数$\gamma$

## 多变量ECM

### 向量形式

对于$k$个I(1)变量$y_t$（k×1向量）：

$\Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta y_{t-i} + \epsilon_t$

其中：
- $\Pi = \alpha \beta'$是影响矩阵
- $\alpha$是调整系数矩阵
- $\beta$是协整向量矩阵
- $\Gamma_i$是短期动态系数矩阵

### 协整矩阵

$\Pi = \alpha \beta'$

其中：
- $\alpha$（k×r）：调整速度矩阵
- $\beta'$（r×k）：协整向量矩阵
- $r$是协整关系的个数（$\Pi$的秩）

### 分解

$\Pi y_{t-1} = \alpha (\beta' y_{t-1}) = \alpha \epsilon_{t-1}$

其中$\epsilon_{t-1} = \beta' y_{t-1}$是r×1的协整误差向量。

## Granger表示定理

### 基本内容

如果$k$个I(1)变量存在$r$个协整关系，则存在ECM表示：

$\Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta y_{t-i} + \epsilon_t$

### 意义

1. 长期关系：$\Pi y_{t-1}$体现协整关系
2. 短期动态：$\Gamma_i \Delta y_{t-i}$体现短期调整
3. 结合：同时描述长期和短期

## ECM的解释

### 长期均衡

$\beta' y_t = 0$

描述变量之间的长期均衡关系。

### 短期调整

$\Delta y_t = \alpha \epsilon_{t-1} + \text{其他项}$

描述短期如何向长期均衡调整。

### 调整系数

$\alpha$的元素：

- $\alpha_{ij} \neq 0$：第$i$个变量参与第$j$个协整关系的调整
- $\alpha_{ij} = 0$：第$i$个变量不参与第$j$个协整关系的调整

## 估计ECM

### 两步法

#### 第一步：估计协整关系

$y_t = \beta x_t + \epsilon_t$

#### 第二步：估计ECM

$\Delta y_t = \alpha + \beta \Delta x_t + \gamma \hat{\epsilon}_{t-1} + u_t$

### Johansen方法

直接估计向量ECM：
1. 估计VAR模型
2. 检验协整关系
3. 估计协整向量和调整系数

## 例子：需求和供给

### 协整关系

价格和数量可能存在长期均衡：
$Q_t = \beta P_t + \epsilon_t$

### ECM

$\Delta Q_t = \alpha_1 + \beta_1 \Delta P_t + \gamma_1 \epsilon_{t-1} + u_t$

$\Delta P_t = \alpha_2 + \beta_2 \Delta Q_t + \gamma_2 \epsilon_{t-1} + v_t$

### 经济含义

- $\gamma_1$：供给向均衡的调整
- $\gamma_2$：需求向均衡的调整
- 一个为正，一个为负（都向均衡调整）

## ECM的优点

### 1. 结合长短期

- 同时描述长期均衡和短期动态
- 更完整的系统描述

### 2. 理论一致性

- 与经济理论一致
- 体现调整机制

### 3. 统计有效性

- 避免伪回归问题
- 参数估计有效

### 4. 预测能力

- 利用长期和短期信息
- 预测更准确

## 与协整关系的关系

### 协整定义

$\epsilon_t = y_t - \beta x_t$是平稳的（I(0)）。

### ECM体现

$\epsilon_{t-1}$项体现偏离均衡的纠正。

### 协整误差

协整误差在ECM中：
- 代表长期关系的偏离
- 驱动短期调整

## 应用

### 1. 宏观经济

- 总需求和总供给
- 货币需求和货币供给
- 利率和汇率

### 2. 金融市场

- 股价和指数
- 利率和债券价格
- 不同市场价格

### 3. 国际贸易

- 进出口关系
- 汇率和贸易平衡

## 检验和诊断

### 1. 协整检验

- EG两步检验
- Johansen检验

### 2. 误差纠正系数检验

检验$\alpha$的元素：
- 检验是否显著（不为0）
- 确认调整机制存在

### 3. 稳定性

检查ECM的稳定性：
- 特征值在单位圆内
- 动态稳定

## 注意事项

### 1. 协整检验

在构建ECM之前先检验协整关系。

### 2. 滞后阶数

ECM中差分项的滞后阶数需要适当选择。

### 3. 理论基础

协整关系应有经济理论支持。

### 4. 稳定性

误差纠正系数应确保系统稳定。

相关链接: [[协整]], [[ADF检验]], [[PP检验]], [[Johansen检验]]

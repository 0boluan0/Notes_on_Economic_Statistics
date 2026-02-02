---
aliases:
- CO迭代
- Cochrane-Orcutt Iteration
- Cochrane
- Cochrane-Orcutt
tags:
- system
- 计量经济学
---
# Cochrane-Orcutt 迭代

## 诊断目的

在存在一阶自相关（AR(1)）的情况下，通过迭代估计自相关系数并变换数据，获得BLUE（最佳线性无偏）估计量。

## 计算方法

### 迭代步骤

1. **初始OLS估计**：
   $$y_t = \beta_0 + \beta_1 x_t + \epsilon_t$$
   得到初始残差 $\hat{\epsilon}_t$

2. **估计自相关系数**：
   $$\hat{\rho} = \frac{\sum_{t=2}^T \hat{\epsilon}_t \hat{\epsilon}_{t-1}}{\sum_{t=2}^T \hat{\epsilon}_{t-1}^2}$$

3. **广义差分变换**：
   $$y_t^* = y_t - \hat{\rho} y_{t-1}$$
   $$x_t^* = x_t - \hat{\rho} x_{t-1}$$

4. **重新估计模型**：
   $$y_t^* = \beta_0(1-\hat{\rho}) + \beta_1 x_t^* + u_t$$

5. **迭代终止判断**：
   - 计算$\hat{\rho}$的新估计值
   - 如果$\hat{\rho}$变化小于阈值，停止
   - 否则返回步骤2

### 收敛标准

| 终止条件 | 常用阈值 |
|----------|----------|
| $\rho$变化 | $|\hat{\rho}_{new} - \hat{\rho}_{old}| < 0.001$ |
| 参数变化 | $|\hat{\beta}_{new} - \hat{\beta}_{old}| < 0.01$ |
| 最大迭代次数 | 50-100次 |

## 估计量性质

| 情况 | CO估计量性质 |
|------|--------------|
| 已知$\rho$ | BLUE |
| 估计$\rho$（迭代充分） | 渐近BLUE |
| 估计$\rho$（迭代不足） | 可能不是BLUE |

## 判断标准

| 指标 | 健康状态 | 问题信号 |
|------|----------|----------|
| 迭代收敛 | ≤10次收敛 | >50次未收敛 |
| $\rho$估计合理 | -1 < $\rho$ < 1 | $\rho$接近边界值 |
| DW检验改善 | DW值接近2 | DW值仍然异常 |

## 常见问题与对策

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 迭代不收敛 | 自相关模式不是AR(1) | 使用更高阶AR模型、Newey-West标准误 |
| $\rho$估计接近1或-1 | 单位根或强自相关 | 检查单位根、考虑差分 |
| 结果不稳定 | 小样本、收敛标准不严格 | 放宽收敛标准、增加迭代次数 |
| 首观测值损失 | 差分变换导致 | 使用Prais-Winsten方法 |

## 替代方法

| 方法 | 适用情况 | 优势 |
|------|----------|------|
| Prais-Winsten | 小样本 | 保留首观测值 |
| Hildreth-Lu | 单变量 |AR(1)精确估计 |
| Durbin两步法 | 快速估计 | 单次变换 |
| Newey-West | 任意自相关 | 非参数方法 |

## 相关概念
[[00_factor/system/Autocorrelation Diagnosis|自相关诊断]]
[[Newey-West]]
[[FGLS]]

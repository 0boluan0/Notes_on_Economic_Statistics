---
aliases:
- ECM
- 误差修正模型
- Error Correction Model
- 误差纠正模型
- 误差纠正机制
- Error Correction Mechanism
tags:
  - 时间序列
  - 计量经济学
---

误差修正模型（Error Correction Model, ECM）是用于描述协整系统短期动态调整的模型，结合了长期均衡关系和短期动态。

## 定义

ECM将变量的一阶差分表示为上一期偏离长期均衡（误差修正项）和其他滞后项的函数。

## 模型形式

$$\Delta y_t = \alpha + \gamma EC_{t-1} + \sum_{i=1}^{p-1} \phi_i \Delta y_{t-i} + \sum_{j=1}^{q-1} \theta_j \Delta x_{t-j} + \varepsilon_t$$

其中：
- EC_{t-1} = y_{t-1} - βx_{t-1}：上一期对长期均衡的偏离
- γ：误差修正系数，调整速度
- γ < 0：表示存在误差修正机制

## 误差修正机制

1. **正向修正**：当y_t高于均衡值（EC_{t-1} > 0），Δy_t倾向为负
2. **负向修正**：当y_t低于均衡值（EC_{t-1} < 0），Δy_t倾向为正
3. **调整速度**：|γ|越大，调整速度越快

## 从协整到ECM

给定协整关系：$y_t = \alpha + \beta x_t + u_t$

ECM形式：
$$\Delta y_t = \gamma u_{t-1} + \sum_{i=1}^{p-1} \phi_i \Delta y_{t-i} + \sum_{j=1}^{q-1} \theta_j \Delta x_{t-j} + \varepsilon_t$$

## 向量ECM（VECM）

对于n维协整系统：

$$\Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta y_{t-i} + \varepsilon_t$$

其中Π = αβ'：
- α：n×r的调整系数矩阵
- β：n×r的协整向量矩阵
- r：协整秩

## 应用

1. **短期动态分析**：分析变量如何向长期均衡调整
2. **政策分析**：评估政策冲击的短期和长期效应
3. **预测模型**：结合长期均衡和短期动态进行预测

相关链接: [[协整]], [[VAR Model|VAR]], [[格兰杰因果检验]]

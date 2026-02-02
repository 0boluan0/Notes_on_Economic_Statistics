---
aliases:
- 脉冲响应函数
- IRF
tags:
- 时间序列
- VAR模型
- concept
---
脉冲响应函数（Impulse Response Function, IRF）用于衡量一个变量的冲击对系统中其他变量的动态影响。

## 定义

脉冲响应函数描述在t=0时刻对某个变量施加一个单位的冲击后，该变量以及系统中其他变量在未来各期的反应。

## 在VAR模型中的应用

对于VAR(p)模型，可以将其转换为VMA(∞)形式：

$$y_t = \sum_{j=0}^{\infty} \Psi_j \varepsilon_{t-j}$$

其中Ψ_j是脉冲响应矩阵，表示第j期后系统对冲击的反应。

## 计算

1. 将VAR模型转换为伴随矩阵形式
2. 通过迭代计算Ψ_j矩阵
3. 绘制脉冲响应图

## 正交化脉冲响应

由于VAR模型中误差项可能相关，需要通过Cholesky分解进行正交化：

$$\varepsilon_t = P u_t$$

其中P是Cholesky分解得到的下三角矩阵，u_t是正交化的冲击。

## 应用

1. **政策分析**：评估货币政策冲击对经济变量的影响
2. **风险分析**：评估风险因子冲击对投资组合的影响
3. **动态乘数**：计算冲击的累积效应

## 解释

脉冲响应图显示：
- 冲击的即时影响（j=0）
- 冲击的持续影响（j>0）
- 冲击的衰减或放大过程
- 冲击的方向（正或负）

相关链接: [[VAR Model|VAR]], [[00_factor/concept/Variance Decomposition|方差分解]], [[00_factor/concept/Granger Causality Test|格兰杰因果检验]]

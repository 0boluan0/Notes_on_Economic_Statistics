---
aliases:
- 偏自相关函数
- PACF
tags:
- 时间序列
- 统计学
- concept
---
偏自相关函数（Partial Autocorrelation Function, PACF）是指在控制中间值影响后，时间序列在特定滞后阶数上的相关程度。

## 定义

偏自相关函数φ_{k,k}是y_t和y_{t-k}在给定中间值y_{t-1}, y_{t-2}, ..., y_{t-k+1}条件下的相关系数。

## 与ACF的区别

- **ACF**：直接衡量y_t和y_{t-k}的相关性，包含中间值的间接影响
- **PACF**：排除中间值影响后，y_t和y_{t-k}的直接相关性

## 在ARMA模型中的应用

- **AR(p)过程**：PACF在p阶后截尾（为0）
- **MA(q)过程**：PACF表现为拖尾（逐渐衰减）
- **ARMA(p,q)过程**：PACF表现为拖尾

## Yule-Walker方程

PACF可以通过求解Yule-Walker方程得到：

$$\begin{pmatrix} \rho_1 \\ \rho_2 \\ \vdots \\ \rho_k \end{pmatrix} = \begin{pmatrix} 1 & \rho_1 & \cdots & \rho_{k-1} \\ \rho_1 & 1 & \cdots & \rho_{k-2} \\ \vdots & \vdots & \ddots & \vdots \\ \rho_{k-1} & \rho_{k-2} & \cdots & 1 \end{pmatrix} \begin{pmatrix} \phi_{k,1} \\ \phi_{k,2} \\ \vdots \\ \phi_{k,k} \end{pmatrix}$$

## 应用

1. 识别AR模型的阶数p
2. 与ACF配合识别ARMA模型

## source_notes

- [[03_平稳时间序列模型#0.回忆用]]（PACF 计算提示）
3. 评估时间序列的自相关结构

相关链接: [[00_factor/concept/Autocorrelation Function|自相关函数]], [[ARMA]]

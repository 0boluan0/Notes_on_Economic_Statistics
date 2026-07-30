---
aliases:
- Ramsey RESET Test
- RESET Test
- Ramsey RESET检验
- 拉姆齐 RESET 检验
tags:
- system
- econometrics
---
# Ramsey RESET Test

## 诊断目标

检查线性回归是否存在函数形式错误或遗漏的非线性结构。

## 什么时候用

怀疑模型漏掉了非线性项、交互项或函数形式设错，但不确定具体漏了什么。

## 检验做法

先估计原模型，得到拟合值 $\hat Y$，再估计辅助回归：

$$
Y_i=X_i'\beta+\gamma_2\hat Y_i^2+\gamma_3\hat Y_i^3+v_i
$$

检验：

$$
H_0:\gamma_2=\gamma_3=0
$$

## 失败模式

- RESET 显著只说明原模型可能设定有问题，不告诉你正确模型。
- 若遗漏变量和拟合值高次项无关，RESET 未必能发现。
- 它不能替代理论建模。

## 来自课程位置

- [[04_模型设定]]

## 关联卡片

## 复现规范

记录输入数据与样本区间、模型/检验设定、阈值或显著性水平、软件版本和处理决策；保留诊断图表与原始输出，使“发现—判断—修正”链条可复核。

- [[Model Misspecification]]
- [[Lagrange Multiplier Test]]
- [[Omitted Variable Bias]]

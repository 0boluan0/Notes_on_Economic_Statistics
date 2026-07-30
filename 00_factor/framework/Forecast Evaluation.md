---
aliases:
- Forecast Evaluation
- Forecast Comparison
- Forecast Accuracy Evaluation
- 预测评估
- 预测效果评估
tags:
- framework
- 时间序列
type: framework
---
# Forecast Evaluation

## 什么时候用

当多个时间序列模型都能拟合历史数据，但你要判断哪个模型**预测未来更可靠**时，用预测评估框架。

## 为什么这样看

样本内拟合好不等于样本外预测好。预测评估的核心是把一段数据留出来，模拟真实预测时“只知道过去、不知道未来”的信息状态。

## 题型识别

看到这些词，进入这张卡：

- out-of-sample forecast；
- holdout sample；
- rolling window / expanding window；
- MSPE；
- Granger-Newbold test；
- Diebold-Mariano test；
- forecast combination。

## 基本判断路线

1. 先设定预测窗口：固定训练集、滚动窗口，或扩展窗口。
2. 对每个模型生成同一预测期的误差 $e_{1i},e_{2i}$。
3. 先比较 MSPE 或 MAE。
4. 如果要做显著性判断，再用 [[Granger-Newbold Test]] 或 [[Diebold-Mariano Test]]。
5. 如果单一模型不稳定，考虑组合预测。

## 边界条件

- 预测误差可能自相关，尤其是多步预测。
- 两个模型的误差往往同期相关，直接用简单 F 比较容易过强。
- 预测评估回答的是预测表现，不自动说明结构因果正确。

## 失败模式

- 用全样本估计后再评价同一全样本预测，造成信息泄漏。
- 只看样本内 AIC/BIC，不看样本外误差。
- 忘记滚动预测时每一期只能使用当时可得信息。

## 来自课程位置

- [[03_平稳时间序列模型#5.5. 预测效果评估|时间序列 03：预测效果评估]]

## 关联卡片

- [[ARMA]]
- [[ARMA Model Identification Steps]]
- [[Granger-Newbold Test]]
- [[Diebold-Mariano Test]]
- [[AIC]]
- [[BIC]]


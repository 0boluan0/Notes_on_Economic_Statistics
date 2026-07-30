---
aliases:
- DID Framework
- Difference-in-Differences framework
- DID framework
- DID使用条件
- 双重差分框架
tags:
- framework
- econometrics
- causal-inference
type: framework
---
# DID Framework

## 什么时候用

用 DID 之前必须同时看到：

- 有处理组和对照组。
- 处理在某个时间点或某些时间点后发生。
- 能观察处理前和处理后结果。
- 有理由相信 [[Parallel Trends]]。

## 为什么有效

DID 不是简单比较政策后水平，而是比较变化。对照组的变化用于扣除共同时间冲击，处理组相对多出来的变化被解释为处理效应。

## 题型识别

| 题目特征 | 判断 |
| --- | --- |
| 两组两期 | 用最基础 DID 均值公式 |
| 多期面板 | 用固定效应或事件研究 |
| 政策前趋势可画 | 必须做平行趋势检查 |
| 不同地区分期处理 | 警惕 TWFE 异质效应偏误 |

## 边界条件

- 处理组和对照组不能有不同的未观测趋势。
- 对照组不能被处理溢出影响。
- 样本构成不能因政策发生系统性变化。
- 不应存在明显预期效应。

## 失败模式

- 只因为有政策前后数据就做 DID。
- 平行趋势图已经分叉仍强行解释。
- 政策同期还有只影响处理组的冲击。
- 错位处理时间下只报告一个 TWFE 系数。

## 来自课程位置

- [[13_面板数据模型]]

## 关联卡片

- [[DID]]
- [[Parallel Trends]]
- [[ATT]]
- [[Fixed Effects Model]]
- [[DID Diagnostics]]

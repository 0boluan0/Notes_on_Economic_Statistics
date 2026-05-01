---
aliases:
- DID Diagnostics
- DID诊断
- DID稳健性
- DID validity checklist
- DID robust checks
tags:
- system
- econometrics
- causal-inference
---
# DID Diagnostics

## 诊断目标

判断 DID 估计是否有可信的因果解释，尤其是 [[Parallel Trends]]、无预期效应、无溢出和样本稳定性。

## 诊断清单

### Step 1：平行趋势

处理前事件研究系数应接近 0，趋势图不应系统性分叉。

### Step 2：预期效应

政策正式实施前不应已经出现处理效应。如果存在提前反应，政策时间定义或识别假设有问题。

### Step 3：溢出效应

对照组不能被政策间接影响。若存在跨地区流动、价格传导或行为替代，对照组不再是反事实。

### Step 4：样本组成

处理前后样本进入退出、权重和测量口径要稳定。

### Step 5：稳健性

- 改时间窗口。
- 改对照组。
- 安慰剂政策时间。
- 安慰剂结果变量。
- 加入合理控制变量和趋势项。

## 失败信号

- 政策前系数显著。
- 处理组政策前趋势持续更快或更慢。
- 政策同期有只影响处理组的事件。
- 分期处理和异质效应明显，但只用一个 TWFE 系数。

## 来自课程位置

- [[13_面板数据模型]]

## 关联卡片

- [[DID]]
- [[Parallel Trends]]
- [[DID Estimation Steps]]
- [[DID Framework]]

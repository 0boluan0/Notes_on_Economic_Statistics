---
aliases:
- DID估计步骤
- DID estimation steps
- Difference-in-Differences estimation
- 双重差分步骤
- DID Estimation Steps
- DID
tags:
- procedure
---
# DID Estimation Steps

## 目标
- 在可复现的数据流程中估计 DID 处理效应。

## Step 1
- 明确处理定义（Treat）、处理时间（Post）、结果变量与样本口径。

## Step 2
- 构造变量：$Treat_i$、$Post_t$ 与交互项 $Treat_i$×$Post_t$。

## Step 3
- 画出处理前趋势（均值图或事件研究图），初步检查平行趋势。

## Step 4
- 估计基准 DID：
  - 简单两组两期：OLS 回归或组均值差上加差。
  - 面板数据：加入个体固定效应与时间固定效应。

## Step 5
- 计算稳健标准误（常用按个体聚类）。

## Step 6
- 做关键稳健性：不同时间窗、安慰剂检验、替代对照组。

## 关联卡片
- [[DID Diagnostics]]
- [[DID Identification Proof]]

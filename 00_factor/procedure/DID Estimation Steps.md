---
aliases:
- DID Estimation Steps
- DID estimation steps
- Difference-in-Differences estimation
- DID估计步骤
- 双重差分步骤
tags:
- procedure
- econometrics
- causal-inference
type: procedure
---
# DID Estimation Steps

## 这张卡什么时候用

当研究设计已经满足 DID 基本结构，并要从数据中估计处理效应时使用。

## 输入

- 处理组指示 $Treat_i$。
- 政策后指示 $Post_t$。
- 结果变量 $Y$。
- 个体、时间和样本口径。
- 可选控制变量。

## 输出

- DID 处理效应估计。
- 标准误和聚类口径。
- 平行趋势与稳健性诊断。

## Step 1：定义处理和时间

明确谁被处理、什么时候开始处理、处理是否一次性或分期发生。

## Step 2：构造交互项

$$
D_{it}=Treat_i\times Post_t
$$

## Step 3：先画趋势

画处理组和对照组处理前后的均值趋势，初步判断 [[Parallel Trends]] 是否可信。

## Step 4：估计基准 DID

两组两期：

$$
Y_{it}=\alpha+\beta(Treat_i\times Post_t)+\gamma Treat_i+\delta Post_t+u_{it}
$$

多期面板常用：

$$
Y_{it}=\alpha_i+\lambda_t+\beta D_{it}+X_{it}'\theta+u_{it}
$$

## Step 5：设置标准误

常见做法是按处理分配层级或个体聚类。不要只报告普通 OLS 标准误。

## Step 6：做诊断和稳健性

- 事件研究或处理前系数。
- 安慰剂处理时间。
- 替代对照组。
- 不同时间窗口。
- 样本构成检查。

## 常见错误

- 没有画处理前趋势。
- 控制变量使用政策后受影响变量。
- 聚类层级过低。
- 对分期处理直接套简单 DID。

## 来自课程位置

- [[13_面板数据模型]]

## 关联卡片

- [[DID]]
- [[DID Diagnostics]]
- [[DID Identification Proof]]
- [[DID Writing Template]]

---
aliases:
- DID Writing Template
- DID写作模板
- DID reporting template
- 双重差分写作
tags:
- writing
- econometrics
- causal-inference
---
# DID Writing Template

## 研究设计

本文采用 Difference-in-Differences（DID）设计，比较处理组与对照组在政策实施前后的结果变量变化差异，以估计政策对处理组的平均处理效应。

## 识别假设

DID 的关键识别假设是平行趋势：在没有政策干预的反事实情形下，处理组与对照组的结果变量应具有相同趋势。本文通过处理前趋势图、事件研究系数和安慰剂检验评估该假设。

## 回归方程

基准模型为：

$$
Y_{it}=\alpha_i+\lambda_t+\beta D_{it}+X_{it}'\theta+\varepsilon_{it}
$$

其中 $\alpha_i$ 为个体固定效应，$\lambda_t$ 为时间固定效应，$D_{it}$ 为处理状态变量，$\beta$ 为 DID 估计量。

## 结果解释

$\beta$ 的估计值表示政策实施后，处理组相对对照组的结果变量平均变化。若 $\beta>0$ 且统计显著，可解释为政策使处理组结果变量相对提高。

## 稳健性

本文进一步更换时间窗口、替代对照组、进行安慰剂政策时间检验，并按处理分配层级聚类标准误，以检查估计结果是否稳健。

## 图表说明

表 X 报告基准 DID 回归结果。图 X 展示处理组与对照组在政策前后的结果变量趋势；政策前趋势接近支持平行趋势假设。

## 关联卡片

- [[DID]]
- [[DID Framework]]
- [[DID Diagnostics]]
- [[DID Identification Proof]]

---
aliases:
- DID写作模板
- DID reporting template
- 双重差分写作
- DID
- DID Writing Template
tags:
- writing
---
# DID Writing Template

## 研究设计段落
- 本文采用 Difference-in-Differences（DID）方法，比较处理组与对照组在政策实施前后的变化差异，以识别政策的因果效应。

## 识别假设段落
- DID 的关键假设为平行趋势：若无政策干预，处理组与对照组的结果变量在处理前后的变化趋势应一致。

## 回归方程段落
- 估计方程设定为：
  $$Y_{it}=\alpha+\beta(Treat_i\times Post_t)+\gamma Treat_i+\delta Post_t+X_{it}'\theta+\varepsilon_{it}$$
  其中 $\beta$ 为 DID 估计量，标准误按个体聚类。

## 结果解释段落
- 估计结果显示 $\beta$ 为正且显著，表明政策使处理组结果变量相对对照组提高了 \% 或 \( \Delta \) 个单位。

## 稳健性与安慰剂段落
- 进一步进行事件研究与安慰剂检验，处理前系数不显著，支持平行趋势假设。

## 图表说明模板
- 表 X 报告基准 DID 回归结果；图 X 展示处理前后的趋势对比。

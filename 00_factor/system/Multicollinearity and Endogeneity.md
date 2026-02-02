---
aliases:
- Multicollinearity and Endogeneity
- 多重共线性与内生性
- Multicollinearity
tags:
- system
---
# Multicollinearity and Endogeneity

## 诊断目标

区分“估计不稳定（共线性）”与“系数有偏（内生性）”两类问题，避免误用修正方法。

## 诊断信号

- **共线性**：VIF 高、条件指数高、系数显著性不稳但 $R^2$ 高。
- **内生性**：理论上存在反向因果或遗漏变量；Hausman 检验显著；IV 与误差相关的证据。

## 处理策略

- **共线性**：
  - 合并或删减高度相关变量。
  - 重新构造指标（比率、主成分）。
  - 增加样本或扩展时间跨度。
- **内生性**：
  - 工具变量（IV/2SLS）。
  - 控制函数或固定效应处理不可观测因素。
  - 结构化设计（DID、RDD）。

## 风险点

- 共线性主要影响标准误，不必然导致系数偏误。
- 内生性直接导致系数偏误与推断错误。

## 报告与复现

- 同时报告 VIF/条件指数与内生性检验。
- 说明识别假设与工具变量来源。

## 相关链接

- [[Multicollinearity|多重共线性]]
- [[Endogeneity|内生性]]
- [[Endogeneity Diagnosis|内生性诊断]]
- [[00_factor/system/Variance Inflation Factor|方差膨胀因子]]
- [[00_factor/system/Condition Index|条件指数]]

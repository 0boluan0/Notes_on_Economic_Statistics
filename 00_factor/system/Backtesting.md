---
aliases:
- VaR回溯测试
- 回溯检验
- Backtesting
tags:
- system
- 金融机构与风险管理
---
# 回溯检验

## 诊断目的

验证风险模型（如VaR）的预测准确性，通过比较模型预测与实际损失，识别模型缺陷和潜在低估风险。

## 计算方法

### Kupiec 比例检验（POF）

$LR_{POF} = -2 \ln\left[\frac{(1-p)^{(n-X)}p^X}{(1-\hat{p})^{(n-X)}\hat{p}^X}\right] \sim \chi^2(1)$

其中：
- n：回溯测试天数（通常250天）
- X：实际例外（超VaR）次数
- p：理论例外率（如1%）
- $\hat{p} = X/n$：$观测例外率$

### Christoffersen 独立性检验

$LR_{ind} = -2 \ln\left[\frac{(1-p)^{N_{00}}p^{N_{01}}(1-p)^{N_{10}}p^{N_{11}}}{(1-\hat{p}_0)^{N_{00}} \hat{p}_0^{N_{01}}(1-\hat{p}_1)^{N_{10}} \hat{p}_1^{N_{11}}}\right] \sim \chi^2(1)$

## 判断标准

### 巴塞尔信号灯标准（99% VaR，250天）

| 颜色 | 例外次数 | mc 乘数 | 含义 |
|------|----------|---------|------|
| 绿色 | 0-4 | 3.0 | 模型良好 |
| 黄色 | 5-9 | 3.4-3.5 | 需关注 |
| 红色 | ≥ 10 | 4.0 | 模型显著低估风险 |

### 例外率检验

| 情况 | p值范围 | 判断 |
|------|----------|------|
| 正常 | p > 0.05 | 不拒绝模型 |
| 轻微异常 | 0.01 < p ≤ 0.05 | 模型可能有问题 |
| 严重异常 | p ≤ 0.01 | 拒绝模型，需修正 |

## 常见问题与对策

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 例外次数过多（低估风险） | 波动率估计偏低、厚尾性忽略 | 使用GARCH模型、改用厚尾分布 |
| 例外次数过少（高估风险） | 波动率估计偏高、模型过于保守 | 调整波动率参数，避免浪费资本 |
| 聚束效应（异常成群出现） | 未建模波动聚集性 | 引入时变波动率模型（GARCH/EWMA） |
| 不同市场状态下表现差异大 | 模型未考虑机制转换 | 使用分段参数或机制转换模型 |

## 相关概念
[[VaR]]
[[Bunching|聚束效应]]
[[VaR Standard Error|VaR标准误]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```

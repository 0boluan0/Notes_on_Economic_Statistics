---
aliases:
- Financial Ratio Analysis
- 财务比率分析
tags:
- procedure
---
# Financial Ratio Analysis

## 适用场景

用于诊断企业经营质量、偿债能力与盈利能力，支持投融资与信用评估。

## 所需数据/条件

- 资产负债表、利润表、现金流量表
- 同行业可比公司或历史基准

## 分析步骤

### 步骤 1：明确分析目的

- 侧重流动性、盈利能力、营运效率或偿债能力。

### 步骤 2：计算关键比率

- **流动性**：流动比率、速动比率、现金比率
- **盈利能力**：ROE、ROA、净利率
- **杠杆**：资产负债率、产权比率、利息保障倍数
- **营运效率**：存货周转率、应收账款周转率

### 步骤 3：对标与分解

- 与历史趋势或同业平均对比。
- 必要时用杜邦分析拆解 ROE。

### 步骤 4：形成诊断结论

- 指出优势/风险与驱动因素。

## 输出物

- 比率表 + 诊断结论 + 风险提示。

## 常见误区

- 忽略会计政策差异导致的口径偏差。
- 只看单期指标、不看趋势与周期。

## 相关链接

- [[DuPont Analysis|杜邦分析]]
- [[Current Ratio|流动比率]]
- [[Return on Equity|权益净利率]]

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

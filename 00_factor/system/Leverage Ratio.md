---
aliases:
- 巴塞尔杠杆率
- 杠杆率
- Leverage Ratio
tags:
- system
- 金融机构与风险管理
---
# 杠杆率（Leverage Ratio）

## 诊断目的

限制银行过度杠杆，作为风险加权资本要求的补充，防止模型风险导致资本充足率被低估。

## 计算方法

$\text{杠杆率} = \frac{\text{Tier 1 资本}}{\text{总风险暴露}} \times 100\%$

### 总风险暴露包括

- 表内资产
- 衍生合约风险暴露
- 证券融资交易暴露
- 表外项目

## 监管要求（Basel III）

| 指标 | 最低要求 | 实施时间 |
|------|----------|----------|
| Tier 1 杠杆率 | ≥ 3% | 2018年全面实施 |

### 合格资本

只有符合严格定义的 Tier 1 资本（主要是普通股和留存收益）才能计入分子。

## 判断标准

| 杠杆率水平 | 评价 | 风险含义 |
|------------|------|----------|
| > 6% | 保守 | 资本非常充足 |
| 3-6% | 合理 | 符合监管要求 |
| < 3% | 危险 | 杠杆过高，需补充资本 |

### 比较分析

| 比率 | 关系 |
|------|------|
| 杠杆率 | 不考虑风险权重 |
| 资本充足率 | 考虑风险权重 |
| 两者差异 | 风险权重模型的准确性和完整性 |

## 常见问题与对策

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 杠杆率低于3%但资本充足率高 | RWA计算偏低、模型风险 | 审查风险权重、保守估计暴露 |
| 杠杆率显著低于资本充足率 | 风险权重系统性偏低 | 检查模型风险、提高风险权重保守度 |
| 衍生品暴露计量复杂 | 潜在未来风险不准确 | 使用保守的附加系数、净额结算 |

## 相关概念
[[Basel Capital Adequacy Ratio|巴塞尔资本充足率]]
[[Tier 1 Capital Ratio|Tier 1 Capital]]
[[NSFR]]

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

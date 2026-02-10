---
aliases:
- 回溯检验
- Back-Testing
- Backtesting
tags:
- 风险管理
- 模型验证
- concept
---
回溯检验（Backtesting）是将模型预测的VaR与实际损益数据对比，统计实际损失超过VaR的次数（称为"例外"或"exception"）的频率，以及这些异常是否随机分布，从而评估VaR模型准确性的方法。

## 检验目的

1. 验证VaR模型的预测能力
2. 识别模型是否低估或高估风险
3. 检测模型是否存在系统性偏差

## 检验内容

完整的VaR回溯检验需要考察：

1. **例外频率检验**（Kupiec检验）
   - 检验实际例外发生概率是否与标称概率一致
   - 使用二项分布或似然比检验

2. **序列独立性检验**（Christoffersen检验）
   - 检验超VaR事件在时间上是否独立分布
   - 检测异常聚束（Bunching）现象

3. **综合回溯评价**（条件覆盖率检验）
   - 同时考察异常比例和异常独立性
   - 只有当两者都满足时，模型才能算通过检验

## 例外（Exception）

例外是指实际损失超过VaR预测值的事件。

例外率 = 实际例外次数 / 观测天数

## 监管信号灯标准

巴塞尔委员会提供的信号灯标准（99% VaR，250天样本）：

| 例外次数 | 信号灯 | 含义 |
|----------|---------|------|
| 0-4次 |   绿色 | 模型合理 |
| 5-9次 |   黄色 | 轻度超标，需要关注 |
| ≥10次  |   红色 | 模型显著低估风险，需要整改 |

## 检验步骤

1. **收集数据**：收集VaR预测值和实际损益数据
2. **统计例外**：识别实际损失超过VaR的交易日
3. **频率检验**：使用Kupiec检验判断例外率是否合理
4. **独立性检验**：使用Christoffersen检验判断异常是否聚集
5. **综合判断**：根据两项检验结果评价模型有效性

## 模型改进方向

若模型未通过回溯检验：

1. **例外率过高**：提高波动率（估计，引入更保守分布
2. **异常聚集显著**：引入时变风险因子模型（如GARCH）
3. **系统性偏差**：调整模型参数或更换模型方法

相关链接: [[VaR]], [[VaR Standard Error|VaR标准误]], [[Kupiec Test|Kupiec检验]], [[Christoffersen Test|Christoffersen检验]], [[GARCH]]

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

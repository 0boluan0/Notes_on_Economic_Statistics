---
aliases:
- Leverage Ratio
- 杠杆率
- Basel leverage ratio
tags:
- system
- banking
- regulation
---
# Leverage Ratio

## 诊断目标

杠杆率是不使用风险权重的资本约束，用来防止 RWA 模型低估风险时银行过度扩张资产负债表。

## 公式

$$
Leverage\ Ratio=\frac{Tier\ 1\ Capital}{Total\ Exposure}
$$

总暴露包括表内资产、衍生品暴露、证券融资交易和表外项目。

## 快速阈值

Basel III 基本最低要求常用 3%。

## 诊断流程

1. 先算 [[Tier 1 Capital Ratio|Tier 1 capital]]。
2. 再算总暴露，不使用风险权重。
3. 与最低杠杆率比较。
4. 若资本充足率高但杠杆率低，重点怀疑 RWA 偏低或低权重资产过度扩张。

## 常见风险点

- 低风险权重资产大量堆积，使资本充足率看起来很好但杠杆过高。
- 表外项目和衍生品暴露漏计。
- 只看 RWA 约束，忽略总资产扩张速度。

## 来自课程位置

- [[16_巴塞尔协议]]

## 关联卡片

## 复现规范

记录输入数据与样本区间、模型/检验设定、阈值或显著性水平、软件版本和处理决策；保留诊断图表与原始输出，使“发现—判断—修正”链条可复核。

- [[Basel Capital Adequacy Ratio]]
- [[Risk-Weighted Assets]]
- [[Tier 1 Capital Ratio]]

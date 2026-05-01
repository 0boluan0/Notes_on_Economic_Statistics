---
aliases:
- Capital and Risk Management
- Capital and Risk Management hub
- 资本与风险管理
- 资本监管知识地图
tags:
- hub
- risk-management
- banking
---
# Capital and Risk Management-hub

## 这组卡解决什么

资本与风险管理这一簇回答：银行面对市场、信用、操作等风险时，如何把损失风险转成监管资本、经济资本、压力测试和风险缓释动作。

## 学习路线

1. 先分清三大风险：[[Market Risk]]、[[Credit Risk]]、[[Operational Risk]]。
2. 再看信用风险参数：[[Default Risk]]、[[PD]]、[[LGD]]、[[EAD]]、[[Credit Conversion Factor]]、[[Credit VaR]]。
3. 然后看衍生品交易对手风险：[[CVA]]、[[DVA]]、[[Position]]、[[Netting]]。
4. 再看资本监管：[[Basel Accords]]、[[Cooke Ratio]]、[[Basel Capital Adequacy Ratio]]、[[Risk-Weighted Assets]]、[[Tier 1 Capital Ratio]]、[[Tier 2 Capital Ratio]]、[[Leverage Ratio]]。
5. 最后看压力视角：[[Stressed VaR]]、[[Scenario Analysis]]、[[Stress Testing]]。

## 风险类型入口

- [[Market Risk]]：利率、汇率、股价、商品价格等市场因子不利变化。
- [[Credit Risk]]：借款人或交易对手不能履约。
- [[Operational Risk]]：流程、人员、系统或外部事件导致损失。

## 信用风险主线

- [[PD]]、[[LGD]]、[[EAD]]：预期损失和监管资本的三个基本参数。
- [[Credit Conversion Factor]]：把表外承诺转换成信用等价敞口。
- [[Default Risk]]：信用风险最核心的违约事件。
- [[Credit VaR]]：信用组合损失分布的高分位损失。
- [[CVA]] 与 [[DVA]]：衍生品估值中对交易对手和自身违约风险的调整。

## 资本监管主线

- [[Basel Accords]]：Basel I/II/III 的整体框架。
- [[Cooke Ratio]]：Basel I 资本充足率口径，最低 8%。
- [[Risk-Weighted Assets]]：把资产按风险权重折算成资本分母。
- [[Basel Capital Adequacy Ratio]]：总资本 / RWA 的系统诊断。
- [[Tier 1 Capital Ratio]]、[[Tier 2 Capital Ratio]]：资本质量与分层。
- [[Leverage Ratio]]：不使用风险权重的补充约束。

## 压力测试主线

- [[Stressed VaR]]：用压力时期数据或参数计算 VaR。
- [[Scenario Analysis]]：给定情景下重估损益。
- [[Stress Testing]]：系统化压力测试和资本承受能力诊断。

## 来自课程位置

- [[16_巴塞尔协议]]：Basel III、CET1、资本缓冲、杠杆率、LCR/NSFR。
- [[15_《巴塞尔协议I II》和 偿付能力法案II]]：Cooke Ratio、RWA、IRB、信用等价额。
- [[19_违约风险]]：信用风险、违约风险、交易对手风险。
- [[21_信用在险价值]]：Credit VaR、PD/LGD/EAD、信用组合损失。
- [[22_情景分析和压力测试]]：情景分析、压力测试和压力 VaR。

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

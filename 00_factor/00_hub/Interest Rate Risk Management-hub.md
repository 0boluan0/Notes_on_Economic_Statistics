---
aliases:
- Interest Rate Risk Management
- Interest Rate Risk Management hub
- 利率风险管理
- 利率风险管理知识地图
tags:
- hub
- risk-management
- fixed-income
---
# Interest Rate Risk Management-hub

## 这组卡解决什么

利率风险管理不是只背久期公式，而是回答三件事：利率变动通过哪条路径伤害银行或组合，如何度量价格/净值敏感度，以及如何用工具把敏感度对冲掉。

## 学习路线

1. 先分清风险来源：[[Repricing Risk]]、[[Basis Risk]]、[[Yield Curve Risk]]、[[Implied Option Risk]]。
2. 再学一阶敏感度：[[duration|Duration]]、[[Macaulay Duration]]、[[Modified Duration]]、[[Dollar Duration]]、[[Basis Point Value (BPV)]]。
3. 然后补非线性和曲线形状：[[Convexity]]、[[Curvature]]、[[Effective Duration]]、[[Key Rate Duration]]。
4. 最后做银行资产负债管理和对冲：[[Interest Rate Sensitivity Gap]]、[[Duration Gap]]、[[DV01 Hedge Calculation]]。

## 风险来源

- [[Repricing Risk]]：资产和负债重新定价时点不匹配。
- [[Basis Risk]]：资产利率和负债利率参考的基准不同，调整幅度不同。
- [[Yield Curve Risk]]：收益率曲线非平行移动导致不同期限敞口受损。
- [[Implied Option Risk]]：提前还款、提前支取等嵌入选择权改变现金流。

## 敏感度卡片

- [[duration|Duration]]：总入口，说明久期为什么能近似价格对利率的敏感度。
- [[Macaulay Duration]]：现金流现值加权平均回收时间。
- [[Modified Duration]]：价格对收益率小幅变化的百分比敏感度。
- [[Dollar Duration]]：把修正久期转换成金额风险。
- [[Basis Point Value (BPV)]]：1bp 利率变化对应的金额变化，也常称 DV01/PV01。
- [[Convexity]]：利率大幅变化时的二阶修正。
- [[Effective Duration]]：现金流会随利率变化时，用重估法测敏感度。
- [[Key Rate Duration]]：不同期限点利率分别变化时的局部敏感度。

## 做题流程

- [[Macaulay Duration Calculation]]：给现金流表时，先算现值权重，再算加权平均时间。
- [[Modified Duration Calculation]]：把 Macaulay duration 转成价格敏感度。
- [[DV01 Hedge Calculation]]：用 BPV/DV01 匹配对冲工具规模。

## 来自课程位置

- [[09_利率风险]]：久期、曲率、局部久期、缺口管理和 DV01 对冲题。
- [[14_VaR参数法和模拟法]]：收益率曲线主成分与利率风险因子映射。

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

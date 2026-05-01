---
aliases:
- Option Greeks
- Option Greeks hub
- 期权希腊字母
- 希腊值知识地图
tags:
- hub
- derivatives
- risk-management
---
# Option Greeks-hub

## 这组卡解决什么

Option Greeks 把衍生品价值变化拆成对市场变量的局部敏感度。学习顺序应是：先分清风险因子，再看一阶/二阶近似，最后用于对冲或 VaR 近似。

## 学习路线

1. 先分清产品是否线性：[[Linear Products]]、[[Nonlinear Products]]。
2. 再学五个基本 Greeks：[[Delta]]、[[Gamma]]、[[Vega]]、[[Theta]]、[[Rho]]。
3. 然后看组合近似：[[Delta Approximation]]、[[Delta-Gamma Approximation]]、[[Cornish-Fisher Expansion]]。
4. 最后做实操：[[Greeks Calculation]]、[[Delta Hedging]]、[[Greeks Hedging Map]]。

## 五个 Greeks 各管什么

- [[Delta]]：标的价格变化的一阶敏感度。
- [[Gamma]]：Delta 对标的价格变化的敏感度，二阶价格风险。
- [[Vega]]：隐含波动率变化风险。
- [[Theta]]：时间流逝风险。
- [[Rho]]：无风险利率变化风险。

## 近似与风险报告

- [[Delta Approximation]]：小幅变动、线性产品或近似线性组合。
- [[Delta-Gamma Approximation]]：非线性产品，标的价格变化不够小时。
- [[Cornish-Fisher Expansion]]：分布偏度、峰度显著时，对正态分位数做修正。

## 来自课程位置

- [[08_操作员如何管理风险暴露]]：Delta/Gamma/Vega/Theta/Rho、线性与非线性产品、对冲和敏感度暴露。
- [[14_VaR参数法和模拟法]]：Delta、Delta-Gamma 和非正态近似在 VaR 中的使用。

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

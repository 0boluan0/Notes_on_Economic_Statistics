---
aliases:
- NPV Calculation
- 净现值计算
- 净现值计算步骤
tags:
- procedure
- finance
---
# NPV Calculation

## 输入

- 项目现金流序列 $CF_0,CF_1,\dots,CF_n$。
- 与现金流风险匹配的折现率 $r$。
- 现金流发生时点和周期。

## 输出

- 项目净现值 $NPV$。
- 接受、拒绝或进一步比较的判断。

## Step 1：列出增量现金流

只记录因接受项目而改变的现金流：

- $CF_0$ 通常是初始投资，记为负数。
- $CF_t$ 包括运营现金流、残值、营运资本回收等。
- 沉没成本不进入现金流，机会成本要进入现金流。

## Step 2：统一时点和折现率口径

- 年现金流用年折现率。
- 月现金流用月折现率。
- 名义现金流配名义折现率，实际现金流配实际折现率。

## Step 3：逐期折现

$$
PV_t=\frac{CF_t}{(1+r)^t}
$$

$t=0$ 的现金流不折现，直接保留原值。

## Step 4：求和得到 NPV

$$
NPV=\sum_{t=0}^{n}\frac{CF_t}{(1+r)^t}
$$

## Step 5：作出判断

- $NPV>0$：项目创造价值。
- $NPV=0$：项目刚好达到要求收益率。
- $NPV<0$：项目不达到要求收益率。

## 检查点

- 现金流符号是否统一，流出为负、流入为正。
- 折现率是否反映项目风险，而不是随手拿存款利率。
- 互斥项目比较时优先看 NPV，而不是只看 [[Internal Rate of Return]]。
- 资金受限时再结合 [[Profitability Index]]。

## 常见错误

- 把会计利润当作现金流。
- 忽略营运资本和残值。
- 用年利率折月现金流。
- 对不同规模项目只看 IRR，忽略绝对价值创造。

## 来自课程位置

- [[05_投资项目资本预算]]

## 关联卡片

- [[Net Present Value]]
- [[Capital Budgeting Decision Map]]
- [[Internal Rate of Return]]
- [[IRR Calculation]]
- [[Investment Decisions]]

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

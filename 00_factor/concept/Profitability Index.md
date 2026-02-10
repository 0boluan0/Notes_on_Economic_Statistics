---
aliases:
- 获利指数
- 现值指数
- PI
- Profitability Index
tags:
- 财务管理
- 投资决策
- concept
---
现值指数是指未来现金流入现值与现金流出现值的比率，用于评估投资项目的盈利能力。

## 含义

现值指数是未来现金流入现值与现金流出现值的比率。因为净现值是绝对值指标，对于初始投资额不相等的项目无法作比较，因此用现值指数相对值指标，能够比较初始投资额不相等的项目。

## 计算公式

$现值指数 = \frac{未来现金流入的现值}{未来现金流出的现值}$

## 决策原则

- PI > 1：项目可行
- PI < 1：项目不可行
- PI = 1：项目刚好保本

## 特点

1. **相对指标**：消除了投资额的差异
2. **比较能力**：能够比较初始投资额不相等的项目
3. **效率导向**：反映投资效率

## 局限性

1. **未消除期限差异**：还没有消除项目期限的差异
2. **绝对值信息**：不反映项目的绝对收益大小

## 与净现值的关系

- 净现值 > 0，则现值指数 > 1
- 净现值 = 0，则现值指数 = 1
- 净现值 < 0，则现值指数 < 1

## 应用场景

1. **资本受限决策**：当资本受限时，选择现值指数高的项目
2. **规模不同项目**：比较投资规模不同的项目
3. **投资效率评估**：评估投资效率

## 与其他方法的配合

- [[Net Present Value|净现值法]]：提供绝对收益信息
- [[Internal Rate of Return|内涵报酬率]]：提供收益率信息

## 相关链接
[[Net Present Value|净现值法]]
[[Internal Rate of Return|内涵报酬率]]

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

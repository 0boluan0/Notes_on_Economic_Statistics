---
aliases:
  - 帕累托
  - Pareto
  - Pareto efficiency
  - 帕累托最优
tags:
  - framework
  - economics
type: framework
---

# Pareto efficiency

## 什么时候用

用于判断一个配置是否存在“至少一方变好、无人变差”的改进空间，或描述多目标优化的效率边界。

## 核心直觉

- **帕累托改进**：从配置 $x$ 变到 $y$，所有主体都不变差，且至少一个主体变好。
- **帕累托最优**：不存在进一步的帕累托改进；它表示效率意义上的不可改进，不表示公平或社会福利最大。

## 关键假设

- 能明确比较每个主体在两个配置下的效用或目标值。
- 比较保持可行性约束，并区分效率判断与价值权重、分配公平。

## 边界与失败模式

- 帕累托最优通常不唯一，且可能极不平等；不能直接推出“应该选择哪个”。
- 若目标相互冲突、不可比较或存在外部性，需先说明评价口径和约束。

## 最小判断流程

1. 写出候选配置的可行集合与每个主体的目标值。
2. 找是否存在一个可行配置，使所有目标不下降且至少一个严格上升。
3. 若存在，原配置不是帕累托最优；若不存在，只能称为帕累托有效。

## 关联卡片

- [[Contract Curve]]
- [[Capital Budgeting Decision Map]]

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

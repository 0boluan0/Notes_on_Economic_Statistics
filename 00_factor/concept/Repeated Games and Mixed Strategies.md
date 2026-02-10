---
aliases:
- Repeated Games and Mixed Strategies
- 重复博弈与混合策略
tags:
- concept
---
# Repeated Games and Mixed Strategies

重复博弈强调“同一博弈在多个时期反复发生”，混合策略强调“以概率分配行动以实现均衡”。

## 重复博弈

- 将单期博弈（stage game）重复 $T$ 期或无限期。
- 关键在于历史依赖与惩罚/奖励机制，使合作成为可持续结果。

## 混合策略

- 玩家在可选行动上分配概率。
- 均衡条件是让对手在其可选行动之间“无差异”。

## 最小例子

在石头剪刀布中，混合策略均衡为各行动各以 $1/3$ 的概率出手。

## 相关链接

- [[Repeated Games|重复博弈]]
- [[Mixed Strategy Nash Equilibrium|混合策略纳什均衡]]
- [[Nash Equilibrium|纳什均衡]]

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

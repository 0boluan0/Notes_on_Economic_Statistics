---
aliases: [State Machine, Transition System, 状态机, 转移系统]
tags: [concept, discrete-mathematics, computation]
type: concept
---
# State Machine

状态机由状态集合 $Q$、起始状态集合 $Q_0\subseteq Q$ 与转移关系 $\delta\subseteq Q\times Q$ 组成。一次执行是从起始状态开始、每一步都遵守 $\delta$ 的有限或无限状态序列；某状态可达，表示存在到达它的有限执行。

状态机只描述“允许怎样变化”。证明安全性通常寻找 [[State Machine Invariant|不变量]]；证明终止性通常寻找 [[Ranking Function|排名函数]]。

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

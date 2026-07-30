---
aliases: [Gale-Shapley Algorithm, Deferred Acceptance Algorithm, 延迟接受算法]
tags: [procedure, discrete-mathematics, algorithms]
type: procedure
---
# Gale-Shapley Algorithm

1. 所有 proposer 初始未匹配，且从未向任何人求婚。
2. 任取未匹配且仍有候选人的 proposer，向尚未尝试过的最高偏好对象求婚。
3. receiver 暂时保留已收到求婚中最喜欢的一位，拒绝其余人；若换人，原暂配者恢复未匹配。
4. 重复直到无人能继续求婚。

每对最多发生一次求婚，所以有限时间终止；严格且完整偏好、两侧人数相等时输出完全稳定匹配。结果对 proposer 一侧最优、receiver 一侧最劣（在所有稳定匹配中比较）。

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

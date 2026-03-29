---
aliases:
- Continuous-Time Markov Chain
- CTMC
- 连续时间马尔可夫链
tags:
- concept
---
# Continuous-Time Markov Chain (CTMC)

>[!note] 它是什么
> - 「CTMC」是指时间连续、状态离散且满足马尔可夫性的随机过程。
>
>[!note] 最小可检索信息
> - 定义：时间连续、状态离散且满足马尔可夫性的随机过程。
> - 符号/公式：生成矩阵 $Q=(q_{ij})$ 给出状态转移率。
> - 最小例子：信用评级在连续时间中的迁移。
>
## 关联卡片
- [[Markov Chain]]

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

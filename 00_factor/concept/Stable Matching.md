---
aliases: [Stable Matching, Stable Marriage Problem, 稳定匹配]
tags: [concept, discrete-mathematics, algorithms]
---
# Stable Matching

在两侧参与者各自对另一侧有严格偏好时，matching 是不重复配对的集合。若存在一对未彼此配对的人，却都更喜欢对方胜过当前对象，则称 blocking pair；没有 blocking pair 的完全匹配称 stable matching。

稳定不等于每个人都得到第一选择，也不等于最大权重匹配。构造算法见 [[Gale-Shapley Algorithm]]。

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

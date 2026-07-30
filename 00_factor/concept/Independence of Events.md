---
aliases: [Independence of Events, Probabilistic Independence, 事件独立]
tags: [concept, discrete-mathematics, probability]
---
# Independence of Events

事件 $A,B$ 独立，当且仅当
$$
\Pr(A\cap B)=\Pr(A)\Pr(B).
$$
若相关条件概率有定义，这等价于 $\Pr(A\mid B)=\Pr(A)$。

独立不同于互斥：两个概率均为正的互斥事件不独立。独立也不自动表示因果关系不存在或存在。

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

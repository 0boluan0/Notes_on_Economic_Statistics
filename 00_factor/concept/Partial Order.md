---
aliases: [Partial Order, Poset, 偏序]
tags: [concept, discrete-mathematics]
---
# Partial Order

集合上的偏序关系满足自反、反对称和传递性。配对 $(P,\preceq)$ 称 poset。并非每两个元素都必须可比较；若任意两元素可比，则是全序。

Hasse 图省略自环、传递边和向上箭头，只保留 cover relation。DAG 的反身可达关系给出偏序；反过来，每个有限 poset 都可由其 Hasse DAG 表示。无限 poset 未必能只靠 cover relations 恢复全部次序。

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

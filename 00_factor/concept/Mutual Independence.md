---
aliases: [Mutual Independence, Joint Independence, 相互独立]
tags: [concept, discrete-mathematics, probability]
---
# Mutual Independence

事件 $A_1,\ldots,A_n$ 相互独立，若对每个非空指标集 $S$ 都有
$$
\Pr\left(\bigcap_{i\in S}A_i\right)=\prod_{i\in S}\Pr(A_i).
$$

只检查每一对事件独立得到的是 pairwise independence，不足以推出 mutual independence。三事件情形还必须检查三重交集。

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

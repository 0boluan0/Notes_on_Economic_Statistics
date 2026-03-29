---
aliases:
- Column Space
- 列空间
tags:
- concept
---
# Column Space

>[!note] 它是什么
> - 「Column Space」是指矩阵列向量所张成的子空间。
>
>[!note] 最小可检索信息
> - 定义：矩阵列向量所张成的子空间。
> - 符号/公式：$\mathrm{Col}(A)=\{Ax\,|\,x\in\mathbb{R}^n\}。$
> - 最小例子：2x2矩阵的列空间由两列向量张成。
>
## 关联卡片
- [[Null Space]]
- [[Subspace]]

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

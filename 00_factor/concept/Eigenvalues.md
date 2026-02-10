---
aliases:
- Eigenvalues
- 特征值
tags:
- concept
---
# Eigenvalues

## 它是什么
- $「Eigenvalues」是指满足 Av=\lambda v 的标量。$

## 最小可检索信息
- 定义：$满足 Av=\lambda v 的标量。$
- 符号/公式：$\det(A-\lambda I)=0。$
- 最小例子：对称矩阵的特征值用于PCA。

## 关联卡片
- [[Spectral Decomposition]]
- [[Condition Index]]

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

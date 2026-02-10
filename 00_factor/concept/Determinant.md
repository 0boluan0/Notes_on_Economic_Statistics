---
aliases:
- Determinant
- 行列式
tags:
- concept
---
# Determinant

## 它是什么
- 「Determinant」是指描述线性变换体积缩放与可逆性的标量。

## 最小可检索信息
- 定义：描述线性变换体积缩放与可逆性的标量。
- 符号/公式：$\det(A)$，如
$$
\det\begin{pmatrix}
a & b \\
c & d
\end{pmatrix} = ad - bc。
$$
- 最小例子：二维线性变换面积缩放为 $|\det(A)|$。

## 关联卡片
- [[Linear Algebra-hub]]

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

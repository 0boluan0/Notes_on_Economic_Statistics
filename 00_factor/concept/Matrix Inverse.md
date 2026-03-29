---
aliases:
- Matrix Inverse
- Inverse Matrix
- 矩阵的逆
- 逆矩阵
tags:
- concept
---
# Matrix Inverse

>[!note] 它是什么
> - 「Matrix Inverse」是指与原矩阵相乘得到单位矩阵的矩阵。
>
>[!note] 最小可检索信息
> - 定义：与原矩阵相乘得到单位矩阵的矩阵。
> - 符号/公式：$AA^{-1}=I$。
> - 最小例子：$2 \times 2$ 矩阵的逆
> $$
> A^{-1} = \frac{1}{ad-bc}
> \begin{pmatrix}
> d & -b \\
> -c & a
> \end{pmatrix}
> $$
>
## 关联卡片
- [[Linear system solution structure]]
- [[Matrix rank properties]]

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

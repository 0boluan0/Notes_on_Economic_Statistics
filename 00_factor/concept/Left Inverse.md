---
aliases:
- Left Inverse
- 左逆
tags:
- concept
- 线性代数
---
# Left Inverse

>[!note] 它是什么
> - 「Left Inverse」是指矩阵 B 满足 $BA=I$，也就是从左侧把 A 抵消掉。
>
>[!note] 最小可检索信息
> - 定义：若存在 B 使 $BA=I$，则 B 是 A 的左逆。
> - 符号/公式：满列秩矩阵 A 可有左逆，如 $(A^TA)^{-1}A^T$。
> - 最小例子：高而瘦、列独立矩阵有左逆但通常没有右逆。
>
## 关键性质
- 左逆对应“列独立”。
- 对非方阵而言，左逆与右逆一般不相同。

## 关联卡片
- [[Right Inverse]]
- [[Pseudoinverse]]
- [[Matrix Inverse]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.outlinks, this.file.link)
)
SORT file.mtime DESC
```

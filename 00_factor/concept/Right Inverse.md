---
aliases:
- Right Inverse
- 右逆
tags:
- concept
- 线性代数
---
# Right Inverse

## 它是什么
- 「Right Inverse」是指矩阵 B 满足 $AB=I$，也就是从右侧把 A 抵消掉。

## 最小可检索信息
- 定义：若存在 B 使 $AB=I$，则 B 是 A 的右逆。
- 符号/公式：满行秩矩阵 A 可有右逆，如 $A^T(AA^T)^{-1}$。
- 最小例子：矮而宽、行独立矩阵有右逆但通常没有左逆。

## 关键性质
- 右逆对应“行独立”。
- 非方阵只有在特殊秩条件下才可能有右逆。

## 关联卡片
- [[Left Inverse]]
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

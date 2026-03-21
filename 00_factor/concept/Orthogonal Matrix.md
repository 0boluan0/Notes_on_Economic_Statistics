---
aliases:
- Orthogonal Matrix
- 正交矩阵
tags:
- concept
- 线性代数
---
# Orthogonal Matrix

## 它是什么
- 「Orthogonal Matrix」是指列向量组成标准正交组的方阵。

## 最小可检索信息
- 定义：满足 $Q^TQ=QQ^T=I$ 的矩阵。
- 符号/公式：$Q^{-1}=Q^T$。
- 最小例子：二维旋转矩阵。

## 关键性质
- 正交矩阵保持长度、角度与内积。
- 正交矩阵的列与行都构成标准正交基。

## 关联卡片
- [[Orthogonality]]
- [[Gram-Schmidt Orthogonalization]]
- [[Spectral Decomposition]]

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

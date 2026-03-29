---
aliases:
- Pseudoinverse
- Moore-Penrose Pseudoinverse
- 伪逆
- 广义逆
tags:
- concept
- 线性代数
---
# Pseudoinverse

>[!note] 它是什么
> - 「Pseudoinverse」是指对非方阵或不可逆矩阵推广出来的“最佳逆”，通常指 Moore-Penrose 伪逆。
>
>[!note] 最小可检索信息
> - 定义：给定矩阵 A，伪逆 $A^+$ 给出最小二乘解与最小范数解的统一表达。
> - 符号/公式：若 $A=U\Sigma V^T$，则 $A^+=V\Sigma^+U^T$。
> - 最小例子：当 A 满列秩时，$A^+=(A^TA)^{-1}A^T$。
>
## 关键性质
- 在 A 可逆时，伪逆退化为普通逆矩阵。
- 在最小二乘问题里，$\hat{x}=A^+b$ 是标准写法。

## 关联卡片
- [[Least Squares]]
- [[Singular Value Decomposition]]
- [[Left Inverse]]
- [[Right Inverse]]

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

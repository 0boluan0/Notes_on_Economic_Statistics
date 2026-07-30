---
aliases:
  - Elementary Matrix
  - elementary matrices
  - 初等矩阵
tags:
  - 线性代数
  - concept
---

# Elementary Matrix

## 它是什么

初等矩阵是对单位矩阵执行一次初等行变换得到的矩阵。左乘初等矩阵 $E$，等价于对被乘矩阵执行同一次行变换：

$$
EA=\text{对 }A\text{ 执行对应行变换后的矩阵}.
$$

三类初等矩阵分别对应：交换两行、把一行乘非零标量、把一行的倍数加到另一行。

## 核心性质

每个初等矩阵都可逆，其逆仍是同类型的初等矩阵。若

$$
E_k\cdots E_2E_1A=R,
$$

则 $A$ 与 $R$ 行等价；若 $A$ 为可逆方阵且 $R=I$，便有 $A^{-1}=E_k\cdots E_1$。

## 关联卡片

- [[Row Equivalence]]
- [[Permutation Matrix]]
- [[Matrix Inverse]]
- [[LU Decomposition]]

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

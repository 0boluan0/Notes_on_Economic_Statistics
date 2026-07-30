---
aliases:
  - Gaussian Elimination
  - row elimination
  - 高斯消元
  - 行消元
tags:
  - 线性代数
  - procedure
type: procedure
---

# Gaussian Elimination

## 何时使用

输入线性系统 $Ax=b$ 或矩阵 $A$，输出阶梯形矩阵、主元、秩、自由变量，并在需要时继续得到 RREF。

## Step 1. 写出增广矩阵

解方程时使用 $[A\mid b]$；只分析子空间时使用 $A$。每次行变换必须作用于整行。

## Step 2. 选择当前主元

从尚未处理的行、列中选非零主元。若当前位置为零而下方有非零元素，先交换两行；数值计算中通常选绝对值较大的主元。

## Step 3. 消去主元下方元素

若主元为 $u_{kk}$，对 $i>k$ 使用

$$
R_i\leftarrow R_i-\frac{a_{ik}}{u_{kk}}R_k.
$$

依次向右、向下推进，直到得到行阶梯形。

## Step 4. 读取结构

- 非零主元数是秩；
- 主元列对应主元变量；
- 非主元列对应自由变量；
- 出现 $[0\ \cdots\ 0\mid c]$ 且 $c\ne0$ 时，系统不相容。

## Step 5. 回代或继续化为 RREF

求一个具体解时从最后一个主元向上回代；需要直接读取特殊解或唯一标准形时，把主元化为 $1$ 并消去主元上方元素。

## 输出检查

- 将所得解代回原方程；
- 检查主元数不超过 $\min(m,n)$；
- 找列空间基时回到原矩阵取主元列。

## 关联卡片

- [[Pivot Position]]
- [[Row Equivalence]]
- [[Reduced Row Echelon Form]]
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

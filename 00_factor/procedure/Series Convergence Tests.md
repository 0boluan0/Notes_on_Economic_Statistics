---
aliases: [Series Convergence Tests, Convergence Tests, 级数收敛判别, 收敛判别法]
tags: [procedure, calculus]
---
# Series Convergence Tests

## Step 1：先做必要条件检查

若 $a_n\not\to0$，则 $\sum a_n$ 必发散；若趋零，仍不能据此断言收敛。

## Step 2：识别标准型

先检查几何级数、$p$-级数、望远镜级数和已知 Taylor 级数。

## Step 3：按结构选判别法

- 正项且易比较：直接比较、极限比较或积分判别。
- 含阶乘、指数或 $n$ 次幂：比值或根值判别。
- 交错符号：交错级数判别，并另查绝对收敛。

## Step 4：端点单独检查

幂级数先求收敛半径，再逐个检查区间端点。

## Step 5：写清结论类型

区分绝对收敛、条件收敛和发散，并注明所用判别法的条件。

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```

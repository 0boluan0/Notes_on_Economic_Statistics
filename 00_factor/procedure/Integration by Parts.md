---
aliases: [Integration by Parts, 分部积分]
tags: [procedure, calculus]
---
# Integration by Parts

## 公式

由乘积法则积分得到

$$
\int u\,dv=uv-\int v\,du.
$$

## Step 1：选择 $u$ 与 $dv$

让 $u$ 求导后变简单，并确保 $dv$ 容易积分。

## Step 2：计算 $du$ 与 $v$

保留符号和常数，必要时先处理定积分上下限。

## Step 3：代入并比较复杂度

新积分应更简单；若更复杂，应重新选择或改用其他技巧。

## Step 4：处理循环或递推

若原积分再次出现，把它移到方程一侧求解；重复结构可形成递推公式。

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```

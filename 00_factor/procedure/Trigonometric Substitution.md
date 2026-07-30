---
aliases: [Trigonometric Substitution, Trig Substitution, 三角代换]
tags: [procedure, calculus]
---
# Trigonometric Substitution

## Step 1：识别根式模板

- $\sqrt{a^2-x^2}$：令 $x=a\sin\theta$。
- $\sqrt{a^2+x^2}$：令 $x=a\tan\theta$。
- $\sqrt{x^2-a^2}$：令 $x=a\sec\theta$。

## Step 2：选择角度范围

选取使代换一一对应且根号符号确定的区间，避免错误去掉绝对值。

## Step 3：替换并积分

写出 $dx$，用三角恒等式消去根式。

## Step 4：回代

通过直角三角形或反三角函数把结果完全写回 $x$，并检查原定义域。

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```

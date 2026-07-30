---
aliases: [Halting Problem, 停机问题]
tags: [concept, discrete-mathematics, computability]
type: concept
---
# Halting Problem

停机问题要求判定：给定程序 $P$ 与输入 $x$，执行 $P(x)$ 是否最终停止。不存在一个对所有程序—输入对都停止且总能给出正确答案的算法，因此该判定问题是不可判定的。

“不可判定”不表示任何具体程序都无法分析；它排除的是覆盖全部输入的总正确判定器。标准证明用自指程序对假设判定器的答案取反，与 [[Cantor Diagonal Argument|对角论证]]同构。

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

---
aliases: [Truth Table, 真值表]
tags: [concept, discrete-mathematics, logic]
type: concept
---
# Truth Table

真值表枚举命题公式中全部原子命题的真值组合，并按连接词语义计算复合公式的真值。含 $n$ 个不同原子命题时共有 $2^n$ 行。

最小例子：$P\Rightarrow Q$ 只在 $P$ 真、$Q$ 假的一行取假。两公式在每一行真值都相同，当且仅当它们 [[Logical Equivalence|逻辑等价]]。

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

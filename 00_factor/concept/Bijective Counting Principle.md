---
aliases: [Bijective Counting Principle, Counting by Bijection, 双射计数原理]
tags: [concept, discrete-mathematics, counting]
---
# Bijective Counting Principle

若存在双射 $f:A\to B$，则 $|A|=|B|$。计数时可把难以直接枚举的对象一一对应到结构更简单的编码。

证明双射必须分别检查：每个输入只有一个输出；不同输入不碰撞（injective）；每个目标都有原像（surjective）。

组合恒等式常通过对同一集合建立两种计数或显式双射证明。

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

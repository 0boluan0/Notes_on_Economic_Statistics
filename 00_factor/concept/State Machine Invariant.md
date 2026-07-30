---
aliases: [State Machine Invariant, State Invariant, Loop Invariant, 状态机不变量, 循环不变量]
tags: [concept, discrete-mathematics]
---
# State Machine Invariant

状态机由状态集合、初始状态和允许转移组成。不变量 $I$ 是每个可达状态都满足的谓词。

证明 $I$ 通常只需两件事：初始状态满足 $I$；任意满足 $I$ 的状态经过任意允许转移后仍满足 $I$。这给出安全性结论，但一般不能证明某个满足 $I$ 的状态必然可达。

执行模板见 [[Invariant Proof Procedure]]。

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

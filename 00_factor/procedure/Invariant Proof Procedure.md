---
aliases: [Invariant Proof Procedure, 不变量证明步骤]
tags: [procedure, discrete-mathematics]
type: procedure
---
# Invariant Proof Procedure

1. 精确定义状态、初始集合、转移规则和坏状态。
2. 提出候选谓词 $I(s)$，要求它足够强以排除坏状态。
3. 验证每个初始状态都满足 $I$。
4. 对每一种转移 $s\to s'$，在 $I(s)$ 下证明 $I(s')$。
5. 由归纳得到所有可达状态满足 $I$，再说明为何这排除目标坏状态。

只检查若干运行轨迹不是不变量证明；遗漏一种转移也会使证明失效。

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

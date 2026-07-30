---
aliases: [Induction Proof Procedure, 归纳证明步骤]
tags: [procedure, discrete-mathematics]
type: procedure
---
# Induction Proof Procedure

1. 写清命题 $P(n)$、论域及起点 $n_0$。
2. 验证所有必要的基础情形。
3. 固定任意 $n\ge n_0$，明确写出归纳假设。
4. 只用归纳假设与已知事实推出 $P(n+1)$。
5. 检查归纳步没有改变对象类别或遗漏余数情形。
6. 引用 [[Mathematical Induction]] 得出全称结论。

若 $P(n)$ 不足以支持下一步，应加强命题，而不是偷偷使用未证明的更强假设。

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

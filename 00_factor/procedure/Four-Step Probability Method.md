---
aliases: [Four-Step Probability Method, 四步概率法]
tags: [procedure, discrete-mathematics, probability]
type: procedure
---
# Four-Step Probability Method

1. **Experiment**：明确随机过程及停止规则。
2. **Sample space**：定义互斥且穷尽的 outcomes。
3. **Event**：把目标翻译成 $A\subseteq\Omega$。
4. **Probability**：为每个 outcome 指定概率并求和。

输出前检查概率非负、总和为 $1$，以及是否误把不等可能结果当作等可能。

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

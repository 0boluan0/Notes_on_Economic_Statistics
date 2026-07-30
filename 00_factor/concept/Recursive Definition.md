---
aliases: [Recursive Definition, 递归定义]
tags: [concept, discrete-mathematics]
---
# Recursive Definition

递归定义用基础对象和构造规则指定一个集合或函数，并隐含“没有其他对象”。完整定义包含：

1. base cases；
2. constructor / recursive cases；
3. closure clause。

例如匹配括号串由空串开始，并在已生成的串上使用连接或外包括号规则生成。要证明所有递归对象的性质，使用 [[Structural Induction]]。

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

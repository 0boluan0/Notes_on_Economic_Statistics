---
aliases: [Counting Strategy Selection, 计数方法选择]
tags: [framework, discrete-mathematics, counting]
type: framework
---
# Counting Strategy Selection

| 结构信号 | 首选工具 | 主要风险 |
|---|---|---|
| 互斥分支 | [[Rule of Sum and Product]] 的加法法则 | 重叠造成重复计数 |
| 连续选择步骤 | 乘法法则 | 同一对象有多条生成路径 |
| 两类对象显然等势 | [[Bijective Counting Principle]] | 映射非单射或非满射 |
| 相同对象分配到有序盒 | [[Stars and Bars]] | 忽略正数/非负限制 |
| 至少两个碰撞 | [[Pigeonhole Principle]] | 容器定义错误 |
| 重叠集合并集 | [[Inclusion-Exclusion Principle]] | 截断符号或交集遗漏 |

先定义“一个结果是什么”，再决定顺序是否重要、重复是否允许、分支是否互斥。

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

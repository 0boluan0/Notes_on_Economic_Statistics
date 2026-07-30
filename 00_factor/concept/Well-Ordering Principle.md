---
aliases: [Well-Ordering Principle, WOP, 良序原理]
tags: [concept, discrete-mathematics]
---
# Well-Ordering Principle

良序原理断言：每个非空的非负整数集合都有最小元素。

它常用于最小反例证明：若坏例集合非空，取最小坏例，再由其更小对象构造矛盾。对自然数命题，良序原理与数学归纳法在逻辑强度上等价。

## 边界

“有下界”在实数中不足以保证最小元，例如 $(0,1)$ 有下界 $0$ 但没有最小元素。

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

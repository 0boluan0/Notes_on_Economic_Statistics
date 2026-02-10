---
aliases:
- 修正久期
- Modified Duration
tags:
- concept
---
修正久期公式为：
$$
\text{修正久期} = \frac{\text{马考利久期}}{1 + \frac{y}{m}}
$$
- $y$：年到期收益率（即名义年利率，Annual Yield）
- $m$：每年复利/付息次数

## 相关链接

一般久期：[[duration|久期]]
马考利久期：[[Macaulay Duration|马考利久期]]

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

---
aliases:
- Loss Given Default
- LGD
- 违约损失率
tags:
- concept
---
# Loss Given Default (LGD)

>[!note] 它是什么
> - 「LGD」是指违约后无法收回的损失比例。
>
>[!note] 最小可检索信息
> - 定义：违约后无法收回的损失比例。
> - 符号/公式：$LGD=1-\text{Recovery Rate}。$
> - 最小例子：回收率40%则LGD=60%。
>
## 关联卡片
- [[Risk-Weighted Assets]]

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

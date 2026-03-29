---
aliases:
- Irreducible Markov Chain
- 不可约链
tags:
- concept
---
# Irreducible Markov Chain

>[!note] 它是什么
> - 「Irreducible Markov Chain」是指任意状态之间都可达的马尔可夫链。
>
>[!note] 最小可检索信息
> - 定义：任意状态之间都可达的马尔可夫链。
> - 符号/公式：对任意 $i,j$，存在 $n$ 使 $P^n_{ij}>0$。
> - 最小例子：有限状态且任一状态可互达的链。
>
## 关联卡片
- [[Stationary Distribution]]

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

---
aliases:
- Markov Process
- 马尔可夫过程
tags:
- concept
---
# Markov Process

## 它是什么
- 「Markov Process」是指满足无记忆性（马尔可夫性）的随机过程。

## 最小可检索信息
- 定义：满足无记忆性（马尔可夫性）的随机过程。
- 符号/公式：$P(X_{t+h}|\mathcal{F}_t)=P(X_{t+h}|X_t)。$
- 最小例子：离散马尔可夫链或连续时间马尔可夫过程。

## 关联卡片
- [[Chapman-Kolmogorov equation]]

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

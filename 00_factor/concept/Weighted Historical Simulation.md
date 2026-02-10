---
aliases:
- Weighted Historical Simulation
- 加权历史模拟
tags:
- concept
---
# Weighted Historical Simulation

## 它是什么
- 「Weighted Historical Simulation」是指对历史收益赋予递减权重以估计风险分位的方法。

## 最小可检索信息
- 定义：对历史收益赋予递减权重以估计风险分位的方法。
- 符号/公式：$w_t=(1-\lambda)\lambda^{t-1}，\sum w_t=1。$
- 最小例子：$用\lambda=0.94加权的历史VaR。$

## 关联卡片
- [[Historical Simulation Method]]

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

---
aliases:
- Clustering Test
- Clustered Violations Test
- 聚束效应检验
tags:
- concept
---
# Clustering Test

## 它是什么
- 「Clustering Test」是指检验异常事件（如VaR超越）是否在时间上聚集、违反独立性的检验。

## 最小可检索信息
- 定义：检验异常事件（如VaR超越）是否在时间上聚集、违反独立性的检验。
- 符号/公式：常用 runs test 或独立性 LR 检验。
- 最小例子：判断VaR超越是否集中发生。

## 关联卡片
- [[VaR-hub]]

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

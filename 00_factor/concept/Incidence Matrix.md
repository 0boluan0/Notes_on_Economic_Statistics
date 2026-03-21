---
aliases:
- Incidence Matrix
- 关联矩阵
- 发生矩阵
tags:
- concept
- 线性代数
---
# Incidence Matrix

## 它是什么
- 「Incidence Matrix」是指用来描述图中节点与边关系的矩阵。

## 最小可检索信息
- 定义：每列对应一条边，每行对应一个节点，通常一列里有一个 `+1` 和一个 `-1`。
- 符号/公式：有向图里，边从起点指向终点时用符号记录方向。
- 最小例子：一条从节点 1 指向节点 2 的边，对应列向量可写成 $(1,-1)^T$。

## 关键性质
- incidence matrix 的 column space、nullspace 与图中的流、环、连通结构直接相关。
- 它把网络问题转化为标准的线性代数问题。

## 关联卡片
- [[Column Space]]
- [[Null Space]]
- [[Row Space]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.outlinks, this.file.link)
)
SORT file.mtime DESC
```

---
aliases:
- Markov Matrix
- 马尔可夫矩阵
tags:
- concept
- 线性代数
---
# Markov Matrix

## 它是什么
- 「Markov Matrix」是指用于描述状态转移、且每列元素和为 1 的非负矩阵。

## 最小可检索信息
- 定义：列和为 1 且元素非负的方阵。
- 符号/公式：若 $u_{k+1}=Au_k$ 且 A 是 Markov matrix，则总和守恒。
- 最小例子：人口或概率在多个状态间的转移矩阵。

## 关键性质
- 1 一定是特征值。
- 稳态通常由特征值 1 对应的特征向量给出。

## 关联卡片
- [[Eigenvalues]]
- [[Matrix Exponential]]

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

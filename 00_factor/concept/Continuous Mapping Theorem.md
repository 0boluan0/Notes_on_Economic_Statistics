---
aliases:
- 连续映射定理
- Continuous Mapping Theorem
tags:
- 概率论
- 统计学
- concept
---
若随机向量序列 $X_n \xrightarrow{p} X$ 且 $g(\cdot)$ 连续，则 $g(X_n) \xrightarrow{p} g(X)$（在相应空间中成立）。常用于将矩收敛/概率收敛通过连续变换传递给函数形式。

## 常见用法

- OLS 一致性证明中，由样本矩 $\to$ 总体矩 推得 $(X'X/n)^{-1} \to Q_{XX}^{-1}$；
- 比例、开方、逆等连续变换的极限传递。

## Connections

- 相关：[[Central Limit Theorem|中心极限定理]]、[[Law of Large Numbers|大数定律]]、Slutsky 定理

## $source_notes$

- [[Central Limit Theorem]]（同章引用）
- [[OLS consistency#步骤 4：应用连续映射定理|OLS估计量的一致性]]

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

---
aliases:
- Principal Component Analysis
- PCA
- 主成分分析
tags:
- concept
---
# Principal Component Analysis (PCA)

## 它是什么
- 「PCA」是指通过正交变换提取方差最大方向的降维方法。

## 最小可检索信息
- 定义：通过正交变换提取方差最大方向的降维方法。
- 符号/公式：$协方差矩阵 \Sigma=V\Lambda V'。$
- 最小例子：用前2个主成分解释90%方差。

## 关联卡片
- [[Linear Algebra-hub]]
- [[Factor Analysis]]
- [[Multicollinearity]]

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

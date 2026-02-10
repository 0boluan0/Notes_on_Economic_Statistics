---
aliases:
- Cornish-Fisher Expansion
- Cornish-Fisher展开
- Cornish
tags:
- concept
---
# Cornish-Fisher Expansion

## 它是什么
- 「Cornish-Fisher Expansion」是指用偏度和峰度对正态分位点进行修正的近似方法。

## 最小可检索信息
- 定义：用偏度和峰度对正态分位点进行修正的近似方法。
- 符号/公式：$z_{cf}=z+\frac{1}{6}(z^2-1)S+\frac{1}{24}(z^3-3z)K-\frac{1}{36}(2z^3-5z)S^2。$
- 最小例子：对收益分布的VaR分位点进行偏度修正。

## 关联卡片
- [[Option Greeks-hub]]

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

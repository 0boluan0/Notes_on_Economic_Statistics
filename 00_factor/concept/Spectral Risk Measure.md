---
aliases:
- Spectral Risk Measure
- 光谱风险度量
tags:
- concept
---
# Spectral Risk Measure

## 它是什么
- 「Spectral Risk Measure」是指以对分位损失加权积分定义的风险度量，权重反映风险厌恶。

## 最小可检索信息
- 定义：以对分位损失加权积分定义的风险度量，权重反映风险厌恶。
- 符号/公式：$\rho(X)=\int_0^1 q_p(X)\,\phi(p)\,dp。$
- 最小例子：对尾部分位赋予更高权重的风险度量。

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

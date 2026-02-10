---
aliases:
- Portmanteau Q Test
- Q Test
- Q检验
tags:
- concept
---
# Q Test

## 它是什么
- 「Q Test」是指检验序列是否存在总体自相关的Q统计量检验（Ljung-Box）。

## 最小可检索信息
- 定义：检验序列是否存在总体自相关的Q统计量检验（Ljung-Box）。
- 符号/公式：$Q=n(n+2)\sum_{k=1}^m \frac{\hat\rho_k^2}{n-k}。$
- 最小例子：检验AR模型残差是否为白噪声。

## 关联卡片
- [[Autocorrelation Diagnosis]]
- [[Breusch-Godfrey Test]]
- [[Durbin-Watson Statistic]]

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

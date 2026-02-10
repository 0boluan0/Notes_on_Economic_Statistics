---
aliases:
- 侦测ARCH/GARCH效应
- McLeod–Li Test
- ARCH LM Test
- ARCH
tags:
- system
- 时间序列
- 波动建模
---
用于检测序列或残差是否存在 ARCH/GARCH 效应（条件异方差）。

## 方法

- Engle ARCH–LM 检验：对残差平方序列回归并做 LM 统计。
- McLeod–Li：对残差平方的自相关做 Ljung–Box 型检验。

要点：
- 在均值模型（如 ARMA）拟合后，对残差（或其平方）进行检验；
- 若显著，考虑 [[ARCH]]/[[GARCH]] 等波动模型。

## Connections

- 相关：[[Ljung-Box Test|Ljung-Box检验]]、[[White Noise Test|白噪声检验]]、[[ARCH]]、[[GARCH]]

## $source_notes$

- [[04_波动建模 Modeling Volatility#2.4 侦测ARCH/GARCH效应]]

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

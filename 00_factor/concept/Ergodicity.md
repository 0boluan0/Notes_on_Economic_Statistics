---
aliases:
- Ergodicity
- 各态历经性
tags:
- concept
---

# Ergodicity

## 它是什么

「Ergodicity」是指时间平均等于总体平均的性质。

>[!note] 它是什么
> - 「Ergodicity」是指时间平均等于总体平均的性质。
>
>[!note] 最小可检索信息
> - 定义：时间平均等于总体平均的性质。
> - 符号/公式：$\lim_{T\to\infty}\frac{1}{T}\sum_{t=1}^T X_t=E[X_t]。$
> - 最小例子：平稳且遍历的过程可用单条长序列估计均值。
>
## 最小例子

平稳且遍历的过程可用单条长序列估计均值。

## 关联卡片
- [[Strict-Sense Stationary Process]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Strict-Sense Stationary Process]]。

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

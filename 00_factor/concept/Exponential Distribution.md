---
aliases:
- Exponential Distribution
- 指数分布
tags:
- concept
---

# Exponential Distribution

## 它是什么

「Exponential Distribution」是指泊松过程的等待时间分布。

>[!note] 它是什么
> - 「Exponential Distribution」是指泊松过程的等待时间分布。
>
>[!note] 最小可检索信息
> - 定义：泊松过程的等待时间分布。
> - 符号/公式：$f(x)=\lambda e^{-\lambda x}，x\ge0。$
> - 最小例子：顾客到达间隔时间。
>
## 最小例子

顾客到达间隔时间。

## 关联卡片
- [[Poisson Process]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Poisson Process]]。

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

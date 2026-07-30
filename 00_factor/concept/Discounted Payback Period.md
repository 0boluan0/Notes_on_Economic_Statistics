---
aliases:
- Discounted Payback Period
- 折现回收期
tags:
- concept
---

# Discounted Payback Period

## 它是什么

「Discounted Payback Period」是指以折现现金流累计为零所需的回收期。

>[!note] 它是什么
> - 「Discounted Payback Period」是指以折现现金流累计为零所需的回收期。
>
>[!note] 最小可检索信息
> - 定义：以折现现金流累计为零所需的回收期。
> - 符号/公式：$求最小 T 使 \sum_{t=1}^T \frac{CF_t}{(1+r)^t}\ge I_0。$
> - 最小例子：折现回收期为3.2年。
>
## 最小例子

折现回收期为3.2年。

## 关联卡片
- [[NPV Calculation]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[NPV Calculation]]。

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

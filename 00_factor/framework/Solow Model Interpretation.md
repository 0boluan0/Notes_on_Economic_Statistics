---
aliases:
- Solow Model Interpretation
- Solow Model Framework
- 索罗模型判断框架
- 索罗模型解释框架
tags:
- framework
- economics
---
# Solow Model Interpretation

## 什么时候用

当题目要求解释储蓄率、人口增长率、折旧率或技术进步对长期增长的影响时，用这张框架卡。

## 为什么这样看

Solow 模型的核心是把人均资本变化拆成两边：

$$
\dot{k}=sf(k)-(n+\delta)k
$$

左边看资本是增加还是减少，右边比较实际投资和盈亏平衡投资。

## 题型识别

- “储蓄率提高”：$sf(k)$ 上移，稳态人均资本和人均产出提高，长期人均增长率不变。
- “人口增长率提高”：$(n+\delta)k$ 变陡，稳态人均资本和人均产出降低。
- “折旧率提高”：效果类似人口增长率提高。
- “技术进步”：长期人均产出增长的来源。
- “收敛”：低于稳态时资本积累，高于稳态时资本下降。

## 边界条件

- 储蓄率、人口增长率、折旧率在模型中外生。
- 标准模型没有解释技术进步来自哪里。
- 跨国比较需要承认不同国家可能有不同稳态。

## 失败模式

- 把储蓄率提高说成永久提高人均增长率。
- 只看总产出增长，忽略人均变量。
- 把 Solow 当成结构转型理论；结构变化要接 [[Kuznets Modern Economic Growth Theory]] 或二元经济模型。

## 来自课程位置

- [[06_经济增长理论#3. 新古典索罗增长模型]]

## 关联卡片

- [[Solow Model]]
- [[Solow Steady State Calculation]]
- [[Steady State Analysis]]
- [[Harrod-Domar Model]]
- [[Kuznets Growth Interpretation]]

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

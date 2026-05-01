---
aliases:
- Solow Model
- Solow Growth Model
- Solow-Swan Model
- 新古典增长模型
- 索罗模型
- 索罗增长模型
- 索洛-斯旺模型
tags:
- concept
- economics
---
# Solow Model

## 一句话记忆

索罗模型说：资本积累决定稳态水平，技术进步决定长期人均增长。

## 它是什么

Solow Model 是外生增长模型，用资本积累、人口增长、折旧和技术进步解释长期经济增长。无技术进步的核心方程是：

$$
\dot{k}=sf(k)-(n+\delta)k
$$

其中 $sf(k)$ 是人均投资，$(n+\delta)k$ 是维持人均资本不变所需的盈亏平衡投资。

## 解决什么判断

- 储蓄率、人口增长率、折旧率变化如何影响稳态人均资本。
- 经济为什么会向稳态收敛。
- 为什么单靠资本积累不能带来持续的人均增长。
- 技术进步为什么成为长期增长的关键。

## 最小例子

若 $y=f(k)=k^{1/2}$，稳态条件是：

$$
s k^{1/2}=(n+\delta)k
$$

解出 $k^*$ 后，$y^*=f(k^*)$。

## 易混点

- 储蓄率上升会提高 $k^*$ 和 $y^*$，但无技术进步时长期人均增长率仍回到 0。
- 人口增长率上升会提高总产出增长率，但降低稳态人均资本和人均产出。
- Solow 是机制模型；[[Kuznets Modern Economic Growth Theory]] 更像现代增长事实和结构变化的总结。

## 来自课程位置

- [[06_经济增长理论#3. 新古典索罗增长模型]]

## 关联卡片

- [[Solow Model Interpretation]]
- [[Solow Steady State Calculation]]
- [[Steady State Analysis]]
- [[Harrod-Domar Model]]
- [[Kuznets Modern Economic Growth Theory]]
- [[Growth Theory-hub]]

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

---
aliases:
- Todaro Migration Model
- Todaro Model
- 托达罗人口迁移模型
- 托达罗模型
tags:
- framework
- economics
type: framework
---
# Todaro Migration Model

## 什么时候用

当题目说城市存在失业但农村人口仍持续迁入，或要求解释迁移决策为什么看“期望收入”而不是只看实际工资时，用托达罗模型。

## 为什么这样看

迁移者比较的是城市期望收入和农村收入。城市工资高并不够，还要乘上找到工作的概率。

$$
E(W_u)=pW_u+(1-p)W_{\text{unemployed}}
$$

若城市期望收入高于农村收入，迁移仍会发生。

## 题型识别

- “城市失业仍有迁移”：用期望收入差解释。
- “过度城市化”：城市工资政策或就业概率变化可能诱发过多迁入。
- “与刘易斯比较”：刘易斯偏充分就业和实际工资差，托达罗加入城市失业和就业概率。
- “政策建议”：不能只提高城市工资，还要创造就业、改善农村收入和信息透明。

## 边界条件

- 迁移者能形成对就业概率和收入的预期。
- 城乡收入差、迁移成本、就业概率会影响迁移决策。
- 城市存在失业或非正规就业风险。

## 失败模式

- 只比较城市工资和农村工资，漏掉就业概率。
- 把迁移写成非理性行为。
- 忽略政策提高城市工资可能反而扩大迁移压力。

## 来自课程位置

- [[03_人口迁移理论#3. 托达罗模型]]

## 关联卡片

- [[Dual Economy Model]]
- [[Lewis Dual Sector Model]]
- [[Fei-Ranis Model]]
- [[Population Migration Theory]]
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

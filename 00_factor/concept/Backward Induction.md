---
aliases:
- Backward Induction
- Backward induction
- 逆向归纳
- 逆向归纳法
tags:
- concept
- game-theory
---

# Backward Induction

## 一句话记忆

逆向归纳是从博弈树末端往前倒推每个决策点的最优选择。

## 它是什么

Backward Induction 是求解有限完美信息动态博弈的方法。先看最后一个决策者会怎么选，再把这个结果替代进前一个决策点，直到回到起点。

## 解决什么判断

- 动态博弈中的均衡路径是什么。
- 哪些威胁在真正到达节点时不可信。
- 完美信息有限博弈的 [[Subgame Perfect Nash Equilibrium]] 是什么。

## 最小例子

进入威慑中，如果进入者进入后，在位者选择“不打价格战”收益更高，那么“你进入我就打价格战”的威胁不可信。逆向归纳会先在进入后的节点排除该威胁。

## 易混点

- 逆向归纳适合有限、完美信息的动态博弈。
- 它求的是序贯理性结果，不一定是总 payoff 最高的结果。
- 不完美信息博弈要谨慎，通常需要信念和 [[Perfect Bayesian Equilibrium]]。

## 来自课程位置

- [[06_扩展性博弈#4. 逆向归纳]]

## 关联卡片

- [[Backward Induction Procedure]]
- [[Extensive-form Game]]
- [[Subgame Perfect Nash Equilibrium]]
- [[Subgame]]
- [[Game Theory-hub]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Subgame Perfect Nash Equilibrium]]、[[Perfect Bayesian Equilibrium]]、[[06_扩展性博弈]]、[[Backward Induction Procedure]]、[[Extensive-form Game]]、[[Subgame]]、[[Game Theory-hub]]。

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

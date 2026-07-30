---
aliases:
- Centipede Game
- 千足虫博弈
tags:
- concept
- game-theory
---

# Centipede Game

## 一句话记忆

千足虫博弈展示了有限动态博弈中，逆向归纳可能推出“一开始就停止”的反直觉结果。

## 它是什么

玩家轮流选择 Stop 或 Pass。若某人 Stop，博弈结束；若一直 Pass，收益可能逐步提高。

## 解决什么判断

- 逆向归纳如何从最后节点往前推。
- 为什么有限重复机会不一定支撑合作。
- 为什么完整策略必须写出每个可能节点上的行动。

## 最小例子

最后行动者若 Stop 比 Pass 收益更高，就会 Stop；前一位玩家预期到这一点后也会 Stop。不断倒推，起点玩家立即 Stop。

## 易混点

- 不能只写“Stop”作为策略；若玩家有多个决策点，要写完整行动计划。
- 千足虫的逆向归纳结果不是在描述现实心理，而是在展示模型理性假设的推论。

## 来自课程位置

- [[06_扩展性博弈#5. 千足虫博弈]]

## 关联卡片

- [[Backward Induction]]
- [[Backward Induction Procedure]]
- [[Subgame Perfect Nash Equilibrium]]
- [[Extensive-form Game]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[06_扩展性博弈]]、[[Backward Induction]]、[[Backward Induction Procedure]]、[[Subgame Perfect Nash Equilibrium]]、[[Extensive-form Game]]。

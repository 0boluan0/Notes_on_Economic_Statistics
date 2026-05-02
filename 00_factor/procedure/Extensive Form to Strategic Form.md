---
aliases:
- Extensive Form to Strategic Form
- Strategic Form of an Extensive-form Game
- 扩展式转策略式
tags:
- procedure
- game-theory
---
# Extensive Form to Strategic Form

## 输入

- 一个扩展式博弈树。
- 每个玩家的决策节点、信息集和终端收益。

## 输出

- 对应的策略式博弈矩阵。
- 每个策略组合对应的终端历史和收益。

## Step 1：列玩家

保持原博弈中的玩家集合不变。自然节点不是玩家。

## Step 2：列每个玩家的完整策略

对每个玩家，写出他在每个可能行动信息集上的完整计划。

即使某个节点在某些路径上不会到达，也要写清如果到达会怎么做。

## Step 3：组合策略

把各玩家的完整策略做笛卡尔积，得到所有策略组合。

## Step 4：沿博弈树找终端历史

对每个策略组合，从起点沿着策略指定的行动走到终端历史。

## Step 5：填收益矩阵

把终端历史上的收益填入对应策略组合的格子。

## 检查点

- 不要把 action 当 strategy。
- 信息集内不能写不同动作。
- payoff 只取终端历史的 payoff。

## 常见错误

- 漏写路径外节点上的行动。
- 把自然节点当玩家。
- 用单个行动命名策略，导致策略集合不完整。

## 来自课程位置

- [[06_扩展性博弈#2. 扩展式转策略式]]

## 关联卡片

- [[Extensive-form Game]]
- [[strategic-form Games]]
- [[Information Set]]
- [[Backward Induction Procedure]]

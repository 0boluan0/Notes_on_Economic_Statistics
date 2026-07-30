---
aliases:
- Bayesian Game
- Games of Incomplete Information
- 贝叶斯博弈
- 不完全信息博弈
tags:
- concept
- game-theory
---

# Bayesian Game

## 一句话记忆

贝叶斯博弈是玩家知道自己的类型，但不完全知道别人类型的博弈。

## 它是什么

贝叶斯博弈在普通策略式博弈基础上加入 [[Type (Game Theory)|类型]] 和 [[Belief (Game Theory)|信念]]。

常用五元素：

| 元素 | 含义 |
|---|---|
| 玩家 | 谁在决策 |
| 类型 | 玩家自己的私人信息状态 |
| 行动 | 每个玩家可实际选择的行为 |
| 信念 | 对其他玩家类型分布的判断 |
| 收益 | 行动与类型共同决定的 payoff |

## 解决什么判断

- 玩家不知道对方成本、偏好、能力或类型时如何建模。
- 为什么策略必须说明每种类型下如何行动。
- 如何把不完全信息问题转成普通策略式问题求解。

## 最小例子

贝叶斯版性别博弈中，玩家 2 有两种类型；玩家 2 知道自己的类型，玩家 1 只知道两种类型的概率。

## 易混点

- 类型不是行动；类型是“我处于什么私人信息状态”。
- 贝叶斯博弈中的 strategy 是类型到行动的函数。
- [[Bayesian Nash Equilibrium]] 是均衡概念，不应混在模型定义里。

## 来自课程位置

- [[05_贝叶斯博弈#2. 贝叶斯博弈]]
- [[05_贝叶斯博弈#3. 将五元素转化为三要素]]

## 关联卡片

- [[Bayesian Nash Equilibrium]]
- [[Type (Game Theory)]]
- [[Belief (Game Theory)]]
- [[Perfect Bayesian Equilibrium]]
- [[Game Theory-hub]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Type (Game Theory)]]、[[Belief (Game Theory)]]、[[Bayesian Nash Equilibrium]]、[[05_贝叶斯博弈]]、[[Perfect Bayesian Equilibrium]]、[[Game Theory-hub]]。

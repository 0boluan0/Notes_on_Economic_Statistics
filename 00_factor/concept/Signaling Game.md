---
aliases:
- Signaling Game
- 信号博弈
tags:
- concept
- game-theory
---
# Signaling Game

## 一句话记忆

信号博弈是发送者知道自己的类型，先发信号；接收者看见信号后更新 belief 并行动。

## 它是什么

典型信号博弈包含：

- 自然决定发送者类型。
- 发送者观察类型并选择信号。
- 接收者观察信号但不直接观察类型。
- 接收者根据信号和 belief 选择行动。

## 解决什么判断

- 信号是否揭示类型。
- separating 或 pooling 候选是否是 PBE。
- 接收者的 belief 如何影响发送者是否愿意伪装。

## 最小例子

Beer-Quiche 博弈中，$A$ 的 weak/strong 是类型，Beer/Quiche 是信号，$B$ 的 Fight/Don't Fight 是接收者行动。

## 易混点

- 信号不是类型本身；信号只是类型可能选择的行动。
- PBE 中必须同时写发送者策略、接收者策略和接收者 belief。

## 来自课程位置

- [[07_子博弈不完全信息#4.1. Beer-Quiche 信号博弈]]

## 关联卡片

- [[Perfect Bayesian Equilibrium]]
- [[Separating Equilibrium]]
- [[Pooling Equilibrium]]
- [[Off-path Belief]]
- [[Type (Game Theory)]]

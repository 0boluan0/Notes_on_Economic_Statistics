---
aliases:
- Pooling Equilibrium
- 合并均衡
- 池化均衡
tags:
- concept
- game-theory
---
# Pooling Equilibrium

## 一句话记忆

合并均衡是不同类型发送同一个信号，使接收者无法从路径上信号区分类型。

## 它是什么

在信号博弈中，若所有类型的发送者选择相同信号，接收者在均衡路径上只能依靠先验 belief 判断类型，这就是 pooling equilibrium。

## 解决什么判断

- 信号是否不揭示类型。
- 路径上 belief 是否等于先验更新。
- 路径外 belief 能否阻止某些类型偏离。

## 最小例子

Beer-Quiche 博弈中，weak 和 strong 都选择 Quiche。若 Beer 不在路径上，接收者看到 Beer 后的 belief 是 off-path belief，需要在 PBE 中一起说明。

## 易混点

- Pooling 不是 mixed strategy equilibrium，中文应理解为合并/池化均衡。
- pooling 能否成立通常取决于路径外 belief 如何支持接收者行动。
- 所有类型选同一信号只是候选结构，不自动构成 PBE。

## 来自课程位置

- [[07_子博弈不完全信息#6. 合并均衡]]

## 关联卡片

- [[Separating Equilibrium]]
- [[Off-path Belief]]
- [[Signaling Game]]
- [[Perfect Bayesian Equilibrium]]
- [[Belief (Game Theory)]]

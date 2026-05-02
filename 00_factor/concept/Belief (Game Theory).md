---
aliases:
- Belief
- Belief (Game Theory)
- Beliefs
- 信念
- 博弈论信念
tags:
- concept
- game-theory
---
# Belief (Game Theory)

## 一句话记忆

belief 是玩家在不确定信息下对“自己在哪个节点/对方是什么类型”的概率判断。

## 它是什么

在贝叶斯博弈中，belief 表示对其他玩家类型分布的判断；在动态不完美信息博弈中，belief 表示玩家在信息集上对各个可能历史的概率判断。

## 解决什么判断

- 接收者看到信号后如何更新对类型的判断。
- 非单点信息集上玩家如何选择最优行动。
- PBE 中路径上和路径外行为如何被支持。

## 最小例子

Beer-Quiche 博弈中，$B$ 观察到 Beer 后，需要判断 $A$ 是 weak 的概率是多少。

## 易混点

- belief 不是事实，而是概率判断。
- 均衡路径上的 belief 不能随便写，必须由 Bayes 法则更新。
- 路径外 belief 可以指定，但仍要支持最优反应。

## 来自课程位置

- [[05_贝叶斯博弈#2.5. 信念]]
- [[07_子博弈不完全信息#2. belief 的引入]]

## 关联卡片

- [[Bayesian Game]]
- [[Perfect Bayesian Equilibrium]]
- [[Off-path Belief]]
- [[Signaling Game]]

---
aliases:
- Subgame
- 子博弈
tags:
- concept
- game-theory
---
# Subgame

## 一句话记忆

子博弈是从单点信息集开始、包含后续所有历史且不切割信息集的后续博弈。

## 它是什么

在扩展式博弈中，一个子博弈必须满足：

- 从单点信息集开始。
- 包含该节点之后所有可能历史。
- 不切割任何信息集。

## 解决什么判断

- 哪些部分可以单独拿出来求纳什均衡。
- SPNE 要在哪些子博弈里检查。
- 为什么有些不完美信息博弈除了整体外没有真子博弈。

## 最小例子

进入威慑中，进入者选择 Enter 后，在位者 Fight/Truce 的后续部分是一个子博弈。

## 易混点

- 博弈整体总是自己的子博弈。
- 不能从非单点信息集中的某个节点单独开始子博弈。
- 没有真子博弈时，SPNE 对 NE 的筛选力可能很弱，需要 [[Perfect Bayesian Equilibrium]]。

## 来自课程位置

- [[06_扩展性博弈#8.1. 子博弈]]
- [[07_子博弈不完全信息#1. 没有真子博弈的动态博弈]]

## 关联卡片

- [[Extensive-form Game]]
- [[Information Set]]
- [[Subgame Perfect Nash Equilibrium]]
- [[Backward Induction]]
- [[Game Theory-hub]]

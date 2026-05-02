---
aliases:
- Perfect Bayesian Equilibrium
- PBE
- Perfect-Bayesian Equilibrium
- 精炼贝叶斯均衡
- 完美贝叶斯均衡
tags:
- concept
- game-theory
---
# Perfect Bayesian Equilibrium

## 一句话记忆

PBE 是“策略 + belief”共同满足顺序理性和信念一致性的动态不完全信息均衡。

## 它是什么

Perfect Bayesian Equilibrium 用于扩展式博弈中存在信息集、类型或信号的情形。一个 PBE 不是单独的策略组合，而是：

$$
(s^*,\mu^*),
$$

其中 $s^*$ 是策略，$\mu^*$ 是每个信息集上的 belief。

## 解决什么判断

- 玩家在看不清历史或类型时如何做最优反应。
- 路径上 belief 是否由 Bayes 法则推出。
- 路径外 belief 能否支持均衡中的不偏离。
- 信号博弈中 separating/pooling 候选是否成立。

## 最小例子

Beer-Quiche 信号博弈中，不能只说 weak/strong 吃什么、接收者 fight 不 fight；还必须说明接收者看到 Beer 或 Quiche 后相信发送者是 weak 的概率。

## 易混点

- PBE 答案必须同时写策略和 belief。
- 正概率到达的信息集，belief 必须由 Bayes 法则更新。
- 零概率到达的信息集，belief 可以指定，但指定后仍要检查最优反应。
- 不同教材对 weak PBE、sequential equilibrium 的包含关系细节不同，考试时优先写本课程的条件。

## 来自课程位置

- [[07_子博弈不完全信息#3. 精炼贝叶斯均衡]]
- [[07_子博弈不完全信息#6. 合并均衡]]

## 关联卡片

- [[Checking Perfect Bayesian Equilibrium]]
- [[Belief (Game Theory)]]
- [[Off-path Belief]]
- [[Signaling Game]]
- [[Separating Equilibrium]]
- [[Pooling Equilibrium]]
- [[Bayesian Game]]

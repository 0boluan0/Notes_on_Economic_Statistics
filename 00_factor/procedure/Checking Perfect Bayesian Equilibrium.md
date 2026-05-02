---
aliases:
- Checking Perfect Bayesian Equilibrium
- PBE Checking Procedure
- 精炼贝叶斯均衡检查步骤
tags:
- procedure
- game-theory
---
# Checking Perfect Bayesian Equilibrium

## 输入

- 一个动态不完全信息博弈。
- 候选策略组合 $s^*$。
- 候选 belief $\mu^*$。

## 输出

- 候选 $(s^*,\mu^*)$ 是否构成 PBE。
- 若失败，指出失败在策略最优性还是 belief 一致性。

## Step 1：列完整策略

写出每个玩家在每个可能信息集上的行动。

发送者要按类型写；接收者要按观察到的信号写。

## Step 2：标出路径上信息集

判断在候选策略下哪些信息集以正概率到达，哪些是路径外信息集。

## Step 3：更新路径上 belief

对正概率到达的信息集，用 Bayes 法则计算 belief。

若候选 belief 与 Bayes 更新不一致，候选失败。

## Step 4：指定路径外 belief

对零概率到达的信息集，写出候选 belief。

路径外 belief 可以自由指定，但不能和题目中仍可用的 Bayes 信息冲突。

## Step 5：检查每个信息集的最优反应

给定 belief 和其他玩家策略，检查当前玩家在每个信息集上是否选择了期望收益最高的行动。

## Step 6：检查发送者是否偏离

尤其在 pooling 候选中，检查每个类型是否愿意偏离到路径外信号。

## 检查点

- 不能只写策略，不写 belief。
- 路径上 belief 不能随便设。
- 路径外 belief 要能支持接收者行动。
- 每个类型都要单独检查偏离。

## 常见错误

- 把 pooling equilibrium 写成 mixed strategy equilibrium。
- 只检查均衡路径，不检查路径外信息集。
- 只检查自然人，不逐类型检查。

## 来自课程位置

- [[07_子博弈不完全信息#3. 精炼贝叶斯均衡]]
- [[07_子博弈不完全信息#6.3. 作答格式]]

## 关联卡片

- [[Perfect Bayesian Equilibrium]]
- [[Belief (Game Theory)]]
- [[Off-path Belief]]
- [[Signaling Game]]
- [[Separating Equilibrium]]
- [[Pooling Equilibrium]]

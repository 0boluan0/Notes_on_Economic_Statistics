---
aliases:
- Game Theory-hub
- Game Theory
- 博弈论
- 博弈论知识地图
tags:
- hub
- game-theory
---
# Game Theory-hub

## 这组卡解决什么

博弈论这组卡回答：多个理性参与者相互影响时，如何描述博弈、删除明显不合理策略、找到稳定结果，并判断动态博弈里的威胁是否可信。

## 学习路线

1. 先会描述一个博弈：[[Game Theory]]、[[player]]、[[strategies]]、[[payoffs]]、[[strategic-form Games]]。
2. 再会简化博弈：[[Strictly Dominated Strategy]]、[[Weakly Dominated Strategy]]、[[Rationalisability]]。
3. 然后求静态均衡：[[Nash Equilibrium]]、[[Pure Strategy Nash Equilibrium]]、[[Finding Pure Strategy Nash Equilibria]]、[[Mixed Strategy Nash Equilibrium]]、[[Finding Mixed Strategy Nash Equilibrium]]。
4. 再处理动态博弈：[[Extensive-form Game]]、[[Information Set]]、[[Subgame]]、[[Backward Induction]]、[[Backward Induction Procedure]]、[[Subgame Perfect Nash Equilibrium]]。
5. 最后看常见模型：[[Cournot Competition]]、[[Bertrand Competition]]、[[Repeated Games]]、[[Bayesian Game]]、[[Perfect Bayesian Equilibrium]]。

## 解题入口

- [[Game Theory Problem Solving Map]]：不知道从哪里下手时先看它。
- [[Game Theory]]：博弈论适用问题、假设和边界。
- [[Finding Pure Strategy Nash Equilibria]]：矩阵博弈找纯策略均衡。
- [[Finding Mixed Strategy Nash Equilibrium]]：没有纯策略均衡或题目要求概率时使用。
- [[Backward Induction Procedure]]：博弈树、进入威慑、千足虫等动态博弈使用。

## 核心概念

- [[player]]：谁在决策。
- [[strategies]]：每个玩家可选的完整行动方案。
- [[payoffs]]：每个策略组合下每个玩家得到的收益。
- [[Nash Equilibrium]]：没有人愿意单独偏离的策略组合。
- [[Best-Reply Function]]：给定对方策略时自己的最优反应集合。

## 动态博弈

- [[Extensive-form Game]]：用博弈树表示顺序、历史和信息。
- [[Information Set]]：玩家在决策点知道什么。
- [[Subgame]]：不切割信息集的后续博弈。
- [[Subgame Perfect Nash Equilibrium]]：每个子博弈里都仍是纳什均衡，用来排除不可信威胁。

## 课程笔记入口

- [[01_导论]]
- [[02_初步认识博弈]]
- [[03_策略式博弈]]
- [[04_案例（囚徒困境与纳什均衡）]]
- [[06_扩展性博弈]]
- [[07_子博弈不完全信息]]

## 复习顺序

1. 能写出策略式三要素 $G=(N,(S_i),(u_i))$。
2. 能区分 strategy、action、strategy profile、payoff。
3. 能用下划线法找纯策略 NE。
4. 能用无差异条件解混合策略 NE。
5. 能解释 SPNE 为什么比 NE 更强。
6. 能用逆向归纳排除不可信威胁。

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

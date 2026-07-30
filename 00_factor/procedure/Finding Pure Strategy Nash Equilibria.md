---
aliases:
- Finding Pure Strategy Nash Equilibria
- Pure Strategy Nash Equilibrium Procedure
- 纯策略纳什均衡求解步骤
tags:
- procedure
- game-theory
type: procedure
---
# Finding Pure Strategy Nash Equilibria

## 输入

- 策略式博弈矩阵或收益函数。
- 每个玩家的策略集合。
- 每个策略组合下各玩家 payoff。

## 输出

- 全部纯策略纳什均衡。
- 若没有纯策略均衡，转向混合策略的提示。

## Step 1：固定对手策略

对行玩家：逐列比较行玩家自己的 payoff。

对列玩家：逐行比较列玩家自己的 payoff。

## Step 2：标出最优反应

- 行玩家在每一列中，标出行玩家 payoff 最大的策略。
- 列玩家在每一行中，标出列玩家 payoff 最大的策略。
- 如果并列最大，全部标出。

## Step 3：找共同标记格

同一个格子里所有玩家都被标为最优反应，则该格子是纯策略纳什均衡。

## Step 4：复核单独偏离

对每个候选格子逐个问：

- 行玩家单独换策略会不会更好？
- 列玩家单独换策略会不会更好？

若都不会，就是 [[Pure Strategy Nash Equilibrium]]。

## 检查点

- 只比较同一个玩家的 payoff。
- 允许多个纯策略均衡。
- 没有共同标记格时，不要说没有纳什均衡，只能说没有纯策略纳什均衡。

## 常见错误

- 在同一格里比较两个玩家谁收益高。
- 忘记并列最优反应。
- 把社会最优格子误当成纳什均衡。

## 来自课程位置

- [[03_策略式博弈#6. 如何找纯策略纳什均衡]]

## 关联卡片

- [[Pure Strategy Nash Equilibrium]]
- [[Nash Equilibrium]]
- [[Best-Reply Function]]
- [[Finding Mixed Strategy Nash Equilibrium]]
- [[Game Theory-hub]]

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

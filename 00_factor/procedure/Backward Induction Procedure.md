---
aliases:
- Backward Induction Procedure
- Backward Induction Steps
- 逆向归纳步骤
tags:
- procedure
- game-theory
type: procedure
---
# Backward Induction Procedure

## 输入

- 有限扩展式博弈树。
- 每个终点的 payoff。
- 每个决策节点的行动者。
- 信息集结构。

## 输出

- 逆向归纳得到的策略组合。
- 均衡路径。
- 对应的 [[Subgame Perfect Nash Equilibrium]]。

## Step 1：从最后的决策节点开始

找到最接近终点的决策节点，比较该节点行动者在各分支下的 payoff。

## Step 2：保留最优分支

保留行动者 payoff 最大的分支。若并列，保留所有最优分支。

## Step 3：用结果替代子博弈

把该节点之后的最优结果折叠成一个 payoff，回到前一个决策节点。

## Step 4：继续倒推

重复 Step 1-3，直到回到博弈起点。

## Step 5：写完整策略

不要只写均衡路径。动态博弈的策略要说明每个玩家在每个可能到达的决策点会怎么做。

## 检查点

- 信息集不是单点时，不能随意从其中一个节点单独倒推。
- 写 SPNE 时要确认每个子博弈里的选择都是最优。
- 均衡路径之外的行动也属于策略。

## 常见错误

- 从起点往后猜，而不是从终点倒推。
- 只写实际发生路径，漏掉 off-path 策略。
- 把不可信威胁保留在均衡里。
- 在不完美信息博弈里机械套逆向归纳。

## 来自课程位置

- [[06_扩展性博弈#4. 逆向归纳]]
- [[06_扩展性博弈#8. 子博弈精炼纳什均衡]]

## 关联卡片

- [[Backward Induction]]
- [[Subgame Perfect Nash Equilibrium]]
- [[Extensive-form Game]]
- [[Subgame]]
- [[Information Set]]
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

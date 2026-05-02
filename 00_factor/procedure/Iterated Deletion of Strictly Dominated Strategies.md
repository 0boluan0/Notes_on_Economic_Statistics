---
aliases:
- Iterated Deletion of Strictly Dominated Strategies
- IDSDS
- Iterated Elimination of Strictly Dominated Strategies
- 严格劣势策略迭代删除
tags:
- procedure
- game-theory
---
# Iterated Deletion of Strictly Dominated Strategies

## 输入

- 一个策略式博弈矩阵。
- 每个玩家的策略集合和收益。

## 输出

- 被删除的策略顺序。
- 最终幸存的策略集合。

## Step 1：逐个玩家比较策略

固定其他玩家的每一种剩余策略，比较同一玩家自己的两个策略收益。

若策略 $s_i$ 在所有情形下都严格高于 $s_i'$，则 $s_i'$ 是当前博弈中的严格劣势策略。

## Step 2：删除严格劣势策略

把被严格占优的策略从当前策略集合中删掉。

只删除能被证明严格劣势的策略；不要因为“看起来差”就删。

## Step 3：在剩余博弈中重新比较

删除后得到的是一个新博弈。继续在新博弈中查找严格劣势策略。

## Step 4：直到无法继续删除

当没有任何玩家的任何剩余策略被严格占优时，停止。

## 检查点

- 每次比较必须只看同一个玩家的收益。
- 删除后必须在剩余博弈里重新比较。
- 严格劣势要求所有相关情形都严格更差。

## 常见错误

- 跨玩家比较收益。
- 用已经被删除的策略证明后续删除。
- 把弱劣势当作严格劣势直接删除。

## 来自课程位置

- [[03_策略式博弈#3. 严格劣势策略迭代删除]]

## 关联卡片

- [[Strictly Dominated Strategy]]
- [[Weakly Dominated Strategy]]
- [[Rationalisability]]
- [[Finding Pure Strategy Nash Equilibria]]
- [[Game Theory-hub]]

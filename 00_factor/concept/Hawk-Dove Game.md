---
aliases:
- Hawk-Dove Game
- Chicken Game
- 鹰鸽博弈
tags:
  - concept
  - game-theory
---
# Hawk-Dove Game

## 一句话记忆

鹰鸽博弈是“强硬对软弱有利，但双方都强硬会很糟”的冲突博弈。

## 它是什么

两个玩家争夺资源。Hawk 表示争斗，Dove 表示让步。若一方 Hawk 一方 Dove，Hawk 获得资源；若双方 Hawk，都会承担冲突成本。

## 解决什么判断

- 为什么会出现两个非对称纯策略均衡。
- 为什么冲突成本足够大时，双方都强硬不可持续。
- 为什么混合策略均衡可解释群体中 Hawk/Dove 比例。

## 最小例子

若资源价值 $v=4$，冲突成本 $c=6$：

|  | Hawk | Dove |
|---|---|---|
| **Hawk** | $(-1,-1)$ | $(4,0)$ |
| **Dove** | $(0,4)$ | $(2,2)$ |

纯策略纳什均衡为 $(Hawk,Dove)$ 与 $(Dove,Hawk)$。

## 易混点

- $v<c$ 是鹰鸽博弈的关键；若参数变了，博弈结构也可能变。
- 双方都 Dove 可能看起来温和，但未必是纳什均衡。

## 来自课程位置

- [[03_策略式博弈#5.4. 鹰鸽博弈]]

## 关联卡片

- [[Nash Equilibrium]]
- [[Mixed Strategy Nash Equilibrium]]
- [[Stag Hunt]]
- [[Prisoner's Dilemma]]

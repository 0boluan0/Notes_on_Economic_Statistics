---
aliases:
- Coordinated Attack Game
- Coordinated Attack
- 协同进攻
- 协调进攻博弈
tags:
- concept
- game-theory
---

# Coordinated Attack Game

## 一句话记忆

协同进攻是只有双方一起进攻才值得进攻的协调博弈。

## 它是什么

两位将军选择 Attack 或 Run。单独进攻会付出成本且失败；双方一起进攻则获得收益。

## 解决什么判断

- 为什么 $(A,A)$ 和 $(R,R)$ 都可能是均衡。
- 混合策略均衡中的进攻概率如何由成本收益比决定。
- 相关信号如何把多重均衡筛成阈值策略。

## 最小例子

|  | A | R |
|---|---|---|
| **A** | $(v-c,v-c)$ | $(-c,0)$ |
| **R** | $(0,-c)$ | $(0,0)$ |

混合策略均衡中，进攻概率满足：

$$
x^*=\frac{c}{v}.
$$

## 易混点

- 这是策略互补问题，不是囚徒困境。
- 成本越高，必须相信对方更可能进攻，自己才愿意进攻。

## 来自课程位置

- [[05_贝叶斯博弈#6. 协同进攻]]
- [[05_贝叶斯博弈#7. 相关类型变体]]

## 关联卡片

- [[Global Game]]
- [[Mixed Strategy Nash Equilibrium]]
- [[Supermodular Game]]
- [[Bayesian Game]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[05_贝叶斯博弈]]、[[Global Game]]、[[Mixed Strategy Nash Equilibrium]]、[[Supermodular Game]]、[[Bayesian Game]]。

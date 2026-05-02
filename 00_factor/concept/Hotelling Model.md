---
aliases:
- Hotelling Model
- Hotelling Competition
- Linear City Model
- Hotelling 模型
tags:
- concept
- game-theory
---
# Hotelling Model

## 一句话记忆

Hotelling 模型是空间差异化下的价格竞争：对手提价会把消费者推向我方。

## 它是什么

两家厂商位于线性城市两端，消费者均匀分布。消费者比较“价格 + 运输成本”，厂商选择价格竞争。

若厂商 $i$ 位于 0 端，厂商 $j$ 位于 1 端，无差异消费者满足：

$$
p_i+t\hat z=p_j+t(1-\hat z).
$$

因此：

$$
\hat z=\frac{t+p_j-p_i}{2t}.
$$

## 解决什么判断

- 产品差异化如何削弱价格竞争。
- 为什么对手提价时，我方最优价格也可能上升。
- Hotelling 为什么和 Cournot 的反应函数方向相反。

## 最小例子

厂商 $i$ 的需求：

$$
q_i=\frac12+\frac{p_j-p_i}{2t}.
$$

利润：

$$
\pi_i=(p_i-c)q_i.
$$

内点最优反应：

$$
B_i(p_j)=\frac{t+c+p_j}{2}.
$$

对称均衡：

$$
p_i^*=p_j^*=c+t.
$$

## 易混点

- $t$ 越大，差异化越强，均衡价格越高。
- 当 $t=0$ 时产品无差异，价格竞争趋近 [[Bertrand Competition]]，不是 [[Cournot Competition]]。
- Hotelling 价格是策略互补；Cournot 产量是策略替代。

## 来自课程位置

- [[04_案例（囚徒困境与纳什均衡）#5. Hotelling 模型]]

## 关联卡片

- [[Oligopoly Competition Map]]
- [[Bertrand Competition]]
- [[Cournot Competition]]
- [[Supermodular Game]]

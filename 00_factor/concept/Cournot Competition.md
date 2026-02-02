---
aliases:
- 古诺竞争
- Cournot Duopoly
- Cournot Competition
- Cournot
tags:
- concept
- game theory
---
# 古诺竞争

## 定义

古诺竞争是静态的寡头竞争模型，厂商针对产量进行竞争。市场价格由总产量线性决定。

## 模型设定

- n 个厂商同时选择产量 q_i
- 市场价格：p = a - Q，其中 Q = q_1 + q_2 + ... + q_n
- 厂商 i 的成本：C_i(q_i)
- 厂商 i 的利润：π_i = p × q_i - C_i(q_i)

## 最优反应函数

对于线性需求 p = a - Q 和边际成本 c 的对称情形：

$$ q_i^* = \max\left\{0, \frac{a - c - Q_{-i}}{2}\right\} $$

其中 Q_{-i} 是其他厂商的总产量。

## 纳什均衡

### 双寡头情形
$$ q_1^* = q_2^* = \frac{a - c}{3} $$
$$ Q^* = \frac{2(a - c)}{3} $$
$$ p^* = \frac{a + 2c}{3} $$

### n 个厂商情形
$$ q_i^* = \frac{a - c}{n + 1} $$
$$ Q^* = \frac{n(a - c)}{n + 1} $$
$$ p^* = \frac{a + nc}{n + 1} $$

## 性质

1. **产量随厂商数量增加**：单个厂商产量下降，但总产量上升

2. **价格收敛于边际成本**：当 n → ∞ 时，p → c（完全竞争）

3. **效率低于完全竞争**：价格高于边际成本，存在无谓损失

## 求解方法

1. 写出利润函数
2. 对自身产量求一阶条件
3. 得到最优反应函数
4. 所有最优反应函数的交点即为纳什均衡

## 与其他模型的比较

### 与 Bertrand 竞争
- Bertrand：价格竞争，均衡是完全竞争价格
- Cournot：产量竞争，价格高于边际成本

### 与 Hotelling 模型
- Cournot 是 Hotelling 在运输成本 t = 0 时的特例
- Hotelling 中，对手提价会增加我的边际收益
- Cournot 中，对手增加产量会降低我的边际收益

## 相关概念

- [[00_factor/concept/Bertrand Competition|Bertrand 竞争]]
- [[00_factor/concept/Hotelling Model|Hotelling 模型]]
- [[00_factor/concept/Best-Reply Function|最优反应函数]]

## 应用

- 石油输出国组织（OPEC）
- 电信行业
- 钢铁行业
